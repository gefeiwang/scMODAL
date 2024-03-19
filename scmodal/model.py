import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scmodal.networks import *
from scmodal.utils import *

class Model(object):
    def __init__(self, batch_size=500, training_steps=10000, seed=1234, n_latent=20,
                 lambdaAE = 10.0, lambdaLA = 10.0, lambdaMNN = 1.0, lambdaGeo = 10.0, lambdaGAN = 1.0, n_KNN = 30,
                 model_path="models", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaAE = lambdaAE
        self.lambdaLA = lambdaLA
        self.lambdaMNN = lambdaMNN
        self.lambdaGeo = lambdaGeo
        self.lambdaGAN = lambdaGAN
        self.n_KNN = n_KNN
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path


    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num
                   ):
        self.adata_A = adata_A_input.copy()
        self.adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = self.adata_A.X
        self.emb_B = self.adata_B.X

    def preprocess_additional_inputs(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num,
                   layer_adata_A_MNN=None, 
                   layer_adata_B_MNN=None, 
                   ):
        # For ATAC-seq data, an option is to let adata_X_input be LSI matrices, 
        # layer_adata_X_MNN be the layer name storing gene activity matrices
        # The first K=shared_gene_num features in self.feat_A_MNN and self.feat_B_MNN should be positively related .

        assert ((layer_adata_A_MNN is not None) or (layer_adata_B_MNN is not None)), "One of the layer names should be feeded; otherwise, use .preprocess() function."
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = adata_A.X
        self.emb_B = adata_B.X
        if layer_adata_A_MNN is None:
            self.feat_A_MNN = self.emb_A
        else:
            self.feat_A_MNN = adata_A.obsm[layer_adata_A_MNN]
        if layer_adata_B_MNN is None:
            self.feat_B_MNN = self.emb_B
        else:
            self.feat_B_MNN = adata_B.obsm[layer_adata_B_MNN]


    def train(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.D_Z = discriminator(self.n_latent).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)
        optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_Z.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)
            K_A = torch.mean((x_A.view(self.batch_size, 1, -1) - x_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A = torch.exp(-K_A/2)
            K_B_z = torch.mean((z_B.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B_z = torch.exp(-K_B_z/2)
            K_B = torch.mean((x_B.view(self.batch_size, 1, -1) - x_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B = torch.exp(-K_B/2)
            K_A_z = torch.mean((z_A.view(self.batch_size, 1, -1) - z_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A_z = torch.exp(-K_A_z/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = (torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            loss_G_GAN = -(torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()

            # geometric structure loss
            loss_Geo = - (torch.clamp(cos(K_A, K_A_z), max=0.975).mean() + torch.clamp(cos(K_B, K_B_z), max=0.975).mean())

            # MNN loss
            Sim = acquire_pairs(self.emb_A[index_A, :self.shared_gene_num], self.emb_B[index_B, :self.shared_gene_num], k=self.n_KNN)
            Sim = torch.from_numpy(Sim).float().to(self.device)
            z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            loss_MNN = torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 2000:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaGeo*loss_Geo, self.lambdaLA*loss_LA, self.lambdaMNN*loss_MNN))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    # def train_additional_inputs(self):
    #     begin_time = time.time()
    #     print("Begining time: ", time.asctime(time.localtime(begin_time)))
    #     self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
    #     self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
    #     self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
    #     self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
    #     self.D_Z = discriminator(self.n_latent).to(self.device)
    #     params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
    #     optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)
    #     optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001)
    #     self.E_A.train()
    #     self.E_B.train()
    #     self.G_A.train()
    #     self.G_B.train()
    #     self.D_Z.train()

    #     N_A = self.emb_A.shape[0]
    #     N_B = self.emb_B.shape[0]

    #     for step in range(self.training_steps):
    #         cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #         index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
    #         index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
    #         x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
    #         x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
    #         z_A = self.E_A(x_A)
    #         z_B = self.E_B(x_B)
    #         x_AtoB = self.G_B(z_A)
    #         x_BtoA = self.G_A(z_B)
    #         x_Arecon = self.G_A(z_A)
    #         x_Brecon = self.G_B(z_B)
    #         z_AtoB = self.E_B(x_AtoB)
    #         z_BtoA = self.E_A(x_BtoA)
    #         K_A = torch.mean((x_A.view(self.batch_size, 1, -1) - x_A.view(1, self.batch_size, -1))**2, dim=2)
    #         K_A = torch.exp(-K_A/2)
    #         K_BtoA = torch.mean((z_B.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
    #         K_BtoA = torch.exp(-K_BtoA/2)
    #         K_B = torch.mean((x_B.view(self.batch_size, 1, -1) - x_B.view(1, self.batch_size, -1))**2, dim=2)
    #         K_B = torch.exp(-K_B/2)
    #         K_AtoB = torch.mean((z_A.view(self.batch_size, 1, -1) - z_A.view(1, self.batch_size, -1))**2, dim=2)
    #         K_AtoB = torch.exp(-K_AtoB/2)

    #         # discriminator loss:
    #         for _ in range(5):
    #             optimizer_D.zero_grad()
    #             loss_D = (torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()
    #             loss_D.backward(retain_graph=True)
    #             optimizer_D.step()

    #         # autoencoder loss:
    #         loss_AE_A = torch.mean((x_Arecon - x_A)**2)
    #         loss_AE_B = torch.mean((x_Brecon - x_B)**2)
    #         loss_AE = loss_AE_A + loss_AE_B

    #         # latent align loss:
    #         loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
    #         loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
    #         loss_LA = loss_LA_AtoB + loss_LA_BtoA

    #         # generator loss
    #         loss_G_GAN = -(torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()

    #         # geometric structure loss
    #         loss_Geo = - (torch.clamp(cos(K_A, K_AtoB), max=0.975).mean() + torch.clamp(cos(K_B, K_BtoA), max=0.975).mean())

    #         # MNN loss
    #         Sim = acquire_pairs(self.feat_A_MNN[index_A, :self.shared_gene_num], self.feat_B_MNN[index_B, :self.shared_gene_num], k=self.n_KNN)
    #         Sim = torch.from_numpy(Sim).float().to(self.device)
    #         z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
    #         loss_MNN = torch.sum(Sim * z_dist) / torch.sum(Sim)

    #         optimizer_G.zero_grad()
    #         loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
    #         loss_G.backward()
    #         torch.nn.utils.clip_grad_norm_(params_G, 5.0)
    #         optimizer_G.step()

    #         if not step % 200:
    #             print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
    #              % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaGeo*loss_Geo, self.lambdaLA*loss_LA, self.lambdaMNN*loss_MNN))

    #     end_time = time.time()
    #     print("Ending time: ", time.asctime(time.localtime(end_time)))
    #     self.train_time = end_time - begin_time
    #     print("Training takes %.2f seconds" % self.train_time)

    #     if not os.path.exists(self.model_path):
    #         os.makedirs(self.model_path)

    #     state = {'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
    #              'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

    #     torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    def eval(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
        self.G_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_A'])
        self.G_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_B'])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

    def get_imputed_df(self, 
                       scale = 'scaled' # if scale=='log', then restore expression after log1p
                       ):

        x_BtoA = self.data_Aspace[self.emb_A.shape[0]:]
        x_AtoB = self.data_Bspace[:self.emb_B.shape[0]]
        if scale == 'log':
            x_BtoA = x_BtoA * self.adata_A.var['std'].values.reshape(1, -1) + self.adata_A.var['mean'].values.reshape(1, -1)
            x_AtoB = x_AtoB * self.adata_B.var['std'].values.reshape(1, -1) + self.adata_B.var['mean'].values.reshape(1, -1)
        imputed_df_BtoA = pd.DataFrame(x_BtoA, index=self.adata_A.obs.index, columns=self.adata_A.var.feature_name)
        imputed_df_BtoA = imputed_df_BtoA.groupby(imputed_df_BtoA.columns, axis=1).mean()
        imputed_df_AtoB = pd.DataFrame(x_AtoB, index=self.adata_B.obs.index, columns=self.adata_B.var.feature_name)
        imputed_df_AtoB = imputed_df_AtoB.groupby(imputed_df_AtoB.columns, axis=1).mean()
        self.imputed_df_BtoA = imputed_df_BtoA
        self.imputed_df_AtoB = imputed_df_AtoB

    def integrate_datasets_links(self, # Use this function for N >= 3 datasets when provided features links for MNN
                                 input_feats,
                                 feat_links_MNN, # A list of index pairs for feature linkages between features in "inputs_MNN"
                                 input_MNN=None, # A list of features matrices for finding MNN pairs between datasets; set as the same as input_feats if "input_MNN=None"
                                 ):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        num_datasets = len(input_feats)
        assert len(feat_links_MNN) == (num_datasets-1)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x_dict = {}
            z_dict = {}
            K_dict = {}
            K_z_dict = {}
            if input_MNN != None:
                assert len(input_MNN) == num_datasets
                x_MNN_dict = {}
            for i in range(num_datasets):
                index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                x_dict[i] = torch.from_numpy(input_feats[i][index_i, :]).float().to(self.device)
                if input_MNN != None:
                    assert input_MNN[i].shape[0] == input_feats[i].shape[0]
                    x_MNN_dict[i] = input_MNN[i][index_i, :]
                z_dict[i] = self.E_dict[i](x_dict[i])
                K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = 0
                for i in range(num_datasets-1):
                    loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE = 0
            for i in range(num_datasets):
                loss_AE += torch.mean((self.G_dict[i](z_dict[i]) - x_dict[i])**2)

            # latent align loss:
            loss_LA = 0
            for i in range(num_datasets-1):
                loss_LA += torch.mean((z_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_dict[i])))**2)
                loss_LA += torch.mean((z_dict[i+1] - self.E_dict[i](self.G_dict[i](z_dict[i+1])))**2)

            # generator loss
            loss_G_GAN = 0
            for i in range(num_datasets-1):
                loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()

            # geometric structure loss
            loss_Geo = 0
            for i in range(num_datasets):
                loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

            # MNN loss
            loss_MNN = 0
            for i in range(num_datasets-1):
                if input_MNN != None:
                    Sim = acquire_pairs(x_MNN_dict[i][:, feat_links_MNN[i][0]], 
                        x_MNN_dict[i+1][:, feat_links_MNN[i][1]], k=self.n_KNN)
                else:
                    Sim = acquire_pairs(x_dict[i][:, feat_links_MNN[i][0]], 
                        x_dict[i+1][:, feat_links_MNN[i][1]], k=self.n_KNN)
                Sim = torch.from_numpy(Sim).float().to(self.device)
                z_dist = torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaGeo*loss_Geo, self.lambdaLA*loss_LA, self.lambdaMNN*loss_MNN))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            z_dict[i] = self.E_dict[i](torch.from_numpy(input_feats[i]).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)


    def integrate_datasets_feats(self, # Use this function for N >= 3 datasets when provided linked features for MNN
                                 input_feats,
                                 paired_input_MNN, # In the form of [[link_feat_data1, link_feat_data2], ..., [link_feat_data(N_1), link_feat_dataN]]
                                 ):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        num_datasets = len(input_feats)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x_dict = {}
            z_dict = {}
            K_dict = {}
            K_z_dict = {}
            assert len(paired_input_MNN) == (num_datasets - 1)
            x_MNN_dict_0 = {}
            x_MNN_dict_1 = {}
            for i in range(num_datasets):
                index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                x_dict[i] = torch.from_numpy(input_feats[i][index_i, :]).float().to(self.device)
                if i < (num_datasets-1):
                    x_MNN_dict_0[i] = paired_input_MNN[i][0][index_i, :]
                if i > 0:
                    x_MNN_dict_1[i-1] = paired_input_MNN[i-1][1][index_i, :]
                z_dict[i] = self.E_dict[i](x_dict[i])
                K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = 0
                for i in range(num_datasets-1):
                    loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE = 0
            for i in range(num_datasets):
                loss_AE += torch.mean((self.G_dict[i](z_dict[i]) - x_dict[i])**2)

            # latent align loss:
            loss_LA = 0
            for i in range(num_datasets-1):
                loss_LA += torch.mean((z_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_dict[i])))**2)
                loss_LA += torch.mean((z_dict[i+1] - self.E_dict[i](self.G_dict[i](z_dict[i+1])))**2)

            # generator loss
            loss_G_GAN = 0
            for i in range(num_datasets-1):
                loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()

            # geometric structure loss
            loss_Geo = 0
            for i in range(num_datasets):
                loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

            # MNN loss
            loss_MNN = 0
            for i in range(num_datasets-1):
                Sim = acquire_pairs(x_MNN_dict_0[i], x_MNN_dict_1[i], k=self.n_KNN)
                Sim = torch.from_numpy(Sim).float().to(self.device)
                z_dist = torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 2000:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaGeo*loss_Geo, self.lambdaLA*loss_LA, self.lambdaMNN*loss_MNN))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            z_dict[i] = self.E_dict[i](torch.from_numpy(input_feats[i]).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)


# Cluster 16

class DCCF_gene(BaseModel):

    def __init__(self, data_handler):
        super(DCCF_gene, self).__init__(data_handler)
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))], dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.intent_num = configs['model']['intent_num']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.is_training = True
        self.final_embeds = False
        usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = torch.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        init(self.user_embeds.weight)
        init(self.item_embeds.weight)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return (G_indices, G_values)

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return (G_indices, G_values)

    def _mask(self):
        embeds = torch.concat([self.user_embeds.weight, self.item_embeds.weight], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def forward(self, masked_user_embeds=None, masked_item_embeds=None):
        if not self.is_training and self.final_embeds is not None:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None)
        if masked_user_embeds is None or masked_item_embeds is None:
            all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        else:
            all_embeds = [torch.concat([masked_user_embeds, masked_item_embeds], dim=0)]
        gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = ([], [], [], [])
        for i in range(0, self.layer_num):
            gnn_layer_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)
            gnn_head_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_h_list)
            gnn_tail_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_t_list)
            int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeds, gnn_tail_embeds)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeds, int_tail_embeds)
            gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            gaa_embeds.append(gaa_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + gaa_layer_embeds + iaa_layer_embeds + all_embeds[i])
        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = all_embeds
        return (user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return (anc_embeds, pos_embeds, neg_embeds)

    def _cal_cl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)
            u_gnn_embs = u_gnn_embs[users]
            u_int_embs = u_int_embs[users]
            u_gaa_embs = u_gaa_embs[users]
            u_iaa_embs = u_iaa_embs[users]
            i_gnn_embs = i_gnn_embs[items]
            i_int_embs = i_int_embs[items]
            i_gaa_embs = i_gaa_embs[items]
            i_iaa_embs = i_iaa_embs[items]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
        return cl_loss

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = self.forward(masked_user_embeds, masked_item_embeds)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)
        recon_loss = self.recon_weight * self._reconstruction(torch.concat([user_embeds, item_embeds], axis=0), seeds)
        loss = bpr_loss + reg_loss + cl_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss}
        return (loss, losses)

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-08) + 1e-08).mean()

class LightGCN_gene(BaseModel):

    def __init__(self, data_handler):
        super(LightGCN_gene, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:])
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return (embeds[:self.user_num], embeds[self.user_num:])

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return (anc_embeds, pos_embeds, neg_embeds)

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate, masked_user_embeds, masked_item_embeds)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds, item_embeds], axis=0), seeds)
        loss = bpr_loss + reg_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss}
        return (loss, losses)

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class AutoCF_gene(BaseModel):

    def __init__(self, data_handler):
        super(AutoCF_gene, self).__init__(data_handler)
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.adj = data_handler.torch_adj
        self.all_one_adj = self.make_all_one_adj()
        self.gt_layer = configs['model']['gt_layer']
        self.gcn_layer = self.hyper_config['gcn_layer']
        self.reg_weight = self.hyper_config['reg_weight']
        self.ssl_reg = self.hyper_config['ssl_reg']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_layer)])
        self.gtLayers = nn.Sequential(*[GTLayer() for i in range(self.gt_layer)])
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()
        usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.gene_masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.gene_masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def make_all_one_adj(self):
        idxs = self.adj._indices()
        vals = t.ones_like(self.adj._values())
        shape = self.adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def get_ego_embeds(self):
        return t.concat([self.user_embeds, self.item_embeds], axis=0)

    def sample_subgraphs(self):
        return self.sampler(self.all_one_adj, self.get_ego_embeds())

    def mask_subgraphs(self, seeds):
        return self.masker(self.adj, seeds)

    def forward(self, encoder_adj, decoder_adj=None, masked_user_embeds=None, masked_item_embeds=None):
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embedsLst = [embeds]
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoder_adj, embedsLst[-1])
            embedsLst.append(embeds)
        if decoder_adj is not None:
            for gt in self.gtLayers:
                embeds = gt(decoder_adj, embedsLst[-1])
                embedsLst.append(embeds)
        embeds = sum(embedsLst)
        return (embeds[:self.user_num], embeds[self.user_num:])

    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = t.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores

    def cal_loss(self, batch_data, encoder_adj, decoder_adj):
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        user_embeds, item_embeds = self.forward(encoder_adj, decoder_adj, masked_user_embeds, masked_item_embeds)
        ancs, poss, _ = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        rec_loss = (-t.sum(anc_embeds * pos_embeds, dim=-1)).mean()
        reg_loss = reg_params(self) * self.reg_weight
        cl_loss = (self.contrast(ancs, user_embeds) + self.contrast(poss, item_embeds)) * self.ssl_reg + self.contrast(ancs, user_embeds, item_embeds)
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds, item_embeds], axis=0), seeds)
        loss = rec_loss + reg_loss + cl_loss + recon_loss
        losses = {'rec_loss': rec_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss}
        return (loss, losses)

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, self.adj)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class GCCF_gene(BaseModel):

    def __init__(self, data_handler):
        super(GCCF_gene, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True
        usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        output_size = int((self.layer_num + 1) * self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(output_size, (self.prf_embeds.shape[1] + output_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + output_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def forward(self, adj=None, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not self.is_training:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None)
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)
        self.final_embeds = embeds
        return (embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1])

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return (anc_embeds, pos_embeds, neg_embeds)

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        user_embeds, item_embeds, _ = self.forward(self.adj, masked_user_embeds, masked_item_embeds)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds, item_embeds], axis=0), seeds)
        loss = bpr_loss + reg_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss}
        return (loss, losses)

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class SGL_gene(LightGCN):

    def __init__(self, data_handler):
        super(SGL_gene, self).__init__(data_handler)
        self.augmentation = configs['model']['augmentation']
        self.node_dropper = NodeDrop()
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def forward(self, adj=None, keep_rate=1.0, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:])
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        if self.augmentation == 'node_drop':
            embeds = self.node_dropper(embeds, keep_rate)
        embeds_list = [embeds]
        if self.augmentation == 'edge_drop':
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(configs['model']['layer_num']):
            random_walk = self.augmentation == 'random_walk'
            tem_adj = adj if not random_walk else self.edge_dropper(tem_adj, keep_rate)
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return (embeds[:self.user_num], embeds[self.user_num:])

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return (anc_embeds, pos_embeds, neg_embeds)

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        keep_rate = configs['model']['keep_rate']
        user_embeds1, item_embeds1 = self.forward(self.adj, keep_rate, masked_user_embeds, masked_item_embeds)
        user_embeds2, item_embeds2 = self.forward(self.adj, keep_rate, masked_user_embeds, masked_item_embeds)
        user_embeds3, item_embeds3 = self.forward(self.adj, 1.0, masked_user_embeds, masked_item_embeds)
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature) + cal_infonce_loss(neg_embeds1, neg_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight
        reg_loss = self.reg_weight * reg_params(self)
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds3, item_embeds3], axis=0), seeds)
        loss = bpr_loss + reg_loss + cl_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss}
        return (loss, losses)

class SimGCL_gene(LightGCN_gene):

    def __init__(self, data_handler):
        super(SimGCL_gene, self).__init__(data_handler)
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.eps = self.hyper_config['eps']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']
        usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.prf_embeds = t.concat([usrprf_embeds, itmprf_embeds], dim=0)
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.mlp = nn.Sequential(nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2), nn.LeakyReLU(), nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1]))
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _mask(self):
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return (masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds)

    def _perturb_embedding(self, embeds):
        noise = F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds) * self.eps
        return embeds + noise

    def forward(self, adj=None, perturb=False, masked_user_embeds=None, masked_item_embeds=None):
        if adj is None:
            adj = self.adj
        if not perturb:
            return super(SimGCL_gene, self).forward(adj, 1.0, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        if masked_user_embeds is None or masked_item_embeds is None:
            embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        else:
            embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return (embeds[:self.user_num], embeds[self.user_num:])

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return (anc_embeds, pos_embeds, neg_embeds)

    def _reconstruction(self, embeds, seeds):
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False, masked_user_embeds=masked_user_embeds, masked_item_embeds=masked_item_embeds)
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight
        recon_loss = self.recon_weight * self._reconstruction(t.concat([user_embeds3, item_embeds3], axis=0), seeds)
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss + cl_loss + recon_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'recon_loss': recon_loss}
        return (loss, losses)

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


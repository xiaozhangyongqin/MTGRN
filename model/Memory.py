import torch
import torch.nn as nn
#
class memeory_augmented(nn.Module):
    def __init__(self, num_nodes=170, mem_num=100, mem_dim=64):
        super(memeory_augmented, self).__init__()
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.memory = self.construct_memory()
        # self.loop_times = loop_times
        self.lamda = 0.7 # fusion fator


    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding


        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict


    def query_memory(self, h_t: torch.Tensor):
        #   h_t.shape: B, T, N, D
        query = h_t
        # first augment
        att_score1 = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value1 = torch.matmul(att_score1, self.memory['Memory'])  # (B, N, d)
        #second augment
        att_score2 = torch.softmax(torch.matmul(value1, self.memory['Memory'].t()), dim=-1)
        value2 = torch.matmul(att_score2, self.memory['Memory'])
       
        # top-2
        _, ind = torch.topk(att_score2, k=2, dim=-1)
        pos = self.memory['Memory'][ind[..., 0]]  # B, N, d
        neg1 = self.memory['Memory'][ind[..., 1]]  # B, N, d
        neg2 = self.memory['Memory'][ind[..., 1]]  # B, N, d
        return self.lamda*value1 + (1-self.lamda)*value2, query, pos, neg1, neg2

    def forward(self, x):
        # x.shape: B,T,N,D
        X_aug, query, pos, neg1, neg2 = self.query_memory(x)

        return X_aug, query, pos, neg1, neg2

if __name__ == '__main__':
    mg = memeory_augmented()
    x = torch.rand(64, 12, 170, 64)
    output, query, pos, neg1, neg2 = mg(x)
    print(output.shape, query.shape, pos.shape, neg1.shape, neg2.shape)




import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


from treelet import treelet_funct

from dataset import train_dataset, get_token_dict_from_file
import json

# unique_tokens_file = "unique_tokens.json"
# def graph_to_index_offline():
#     """ Assign node names such as Var or VarFunc to unique one_hot vectors (where 1 means [0, 1, 0, ..., 0]).
#     This should be done offline and saved as a dictionary format (json.
    
#     @ Output (.json): Dictionary, with {Node name such as "var", "VarFunc": unique one_hot vector}
#     """
#     count = 0 # Number of unique vars found.
#     NUM_TOKENS = 1909
#     node_to_index = {}

#     for datapoint in train_dataset():
#         conjecture = datapoint.conjecture
#         statement = datapoint.statement
#         for _, node in conjecture.nodes.items():
#             token = node.token
#             if token not in node_to_index:
#                 node_to_index[token] = count
#                 count += 1
#                 print("Number of tokens found: "+str(count))
#             if count == NUM_TOKENS:
#                 print("All Unique Tokens Found!")
#                 dumped = json.dumps(node_to_index)
#                 with open(unique_tokens_file, "w") as f:
#                     f.write(dumped)
#                 return


class LinearMap(nn.Module):
    '''
    Map the one-hot vector to the x's corresponding to inputs of each neural network.
    '''
    def __init__(self):
        super(LinearMap, self).__init__()
        INPUT_DIM = 1909
        HL1 = 256
        self.fc1 = nn.Linear(INPUT_DIM, HL1)

    def forward(self, x):
        return self.fc1(x)



class FPClass(nn.Module):
    '''
    FP is the outer function in Section 3.3 used for Order-Preserving Embeddings 
    Input:  Linear combination of the node and neighboring update functions.
    Output: The next value for the node.
    '''
    def __init__(self):
        super(FPClass, self).__init__()
        INPUT_DIM = 256
        HL1 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return x

class FIClass(nn.Module):
    def __init__(self):
        super(FIClass, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        """
        @ Args:
            x_batch (dense vectors, shape = [batch_size, length of parents(xv), xv for each xv)
                Collection of (x_u, x_v) 

                x_batch[:,:, xv_id] = The summands of FI for fixed xv

        @ Output:
            in_sum (array of dense_vectors): Size = Number of Nodes in G
        """
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch


class FOClass(nn.Module):
    def __init__(self):
        super(FOClass, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = HL2 = 256
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        # Order is (batch, xv, xu)
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch


class FHClass(nn.Module):
    def __init__(self):
        super(FHClass, self).__init__()
        INPUT_DIM = 256 * 3
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        # Order in xv, xu, xw
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch

class FLClass(nn.Module):
    def __init__(self):
        super(FLClass, self).__init__()
        INPUT_DIM = 256 * 3
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)


    def forward(self, x_batch):
        # Order in xu, xv, xw
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch


class FRClass(nn.Module):
    def __init__(self):
        super(FRClass, self).__init__()
        INPUT_DIM = 256 * 3
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)


    def forward(self, x_batch):
        # Order: batch, xu, xw, xv
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch



class CondClassifier(nn.Module):
    def __init__(self):
        super(CondClassifier, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = 256
        HL2 = 2
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

        self.fc2 = nn.Linear(HL1, HL2)

    def forward(self, x_conj, x_state):
        # Order in (batch, x_conj, x_state)
        # print(x_conj.shape)
        # print(x_state.shape)
        # print(x_conj.shape)
        x = torch.cat([x_conj, x_state], dim = 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x

    def predict(self, x_conj, x_state): # Only used for testing.
        scores = self.forward(x_conj, x_state)
        return np.argmax(scores)


class max_pool_dense_graph(nn.Module):
    def __init__(self):
        super(max_pool_dense_graph, self).__init__()
        INPUT_DIM = 256 * 2
        self.pool = nn.MaxPool1d(2)
        # self.pool = nn.MaxPool1d(2)

    def forward(self, G_dense):
        x1 = G_dense[0]
        x1.unsqueeze_(0)
        for x2 in G_dense:
            x2.unsqueeze_(0)
            x = torch.stack([x1, x2], dim = 2)

            # print(x1)
            # print(x2)
            # print(x)

            x1 = self.pool(x)
            x1.squeeze_(2)
        x1.squeeze_(0)
        return x1


class FormulaNet(nn.Module):
    def __init__(self, num_steps, cuda_available = False):
        super(FormulaNet, self).__init__()
        # Initialize models
        self.dense_map = LinearMap() # maps one_hot -> 256 dimension vector
        self.FP = FPClass()
        self.FI = FIClass()
        self.FO = FOClass()
        self.FL = FLClass()
        self.FR = FRClass()
        self.FH = FHClass()
        self.Classifier = CondClassifier()
        self.Softmax = nn.Softmax(dim = 1)

        self.max_pool_dense_graph = max_pool_dense_graph()

        self.num_steps = num_steps
        self.token_to_index = get_token_dict_from_file()

        self.cuda_available = cuda_available


    # Given a graph and all the functions, do one update in parallel for all nodes in the graph.
    def fullPass(self, dense_nodes, G):
        """
        @ Args:
            <F?> (Neural Network object)
            dense_nodes (2D Array): 0-axis indexes the node index. 
                dense_nodes[i] is the 256 dimensional dense representation of the ith node, 
                where i is the node's unique id in <Graph> object
            G (<Graph> object): Graph object, necessary to iterate over edges and trelets

        @ Vars:
            in_batch (list): Whatever will be fed into FI, FO, ... FR
            dv (Tensor): dv[xv] for each xv
            ev (Tensor): ev[xv] for each xv
            <Neural Function>_indices (dict): {xv: [index of in_batch]} to be used to for summands in update equation (indices of in_sum).
            new_<Neural Function>_sum (Tensor): in_sum, out_sum, etc. collapsed such that new_sum[xv] = sum_{xu in parent(xv)} F_I (xu, xv)
                shape: (num_nodes, 256)

        @ Return:
            new_nodes (Same shape as Mdense_nodes>): One-step update of <dense_nodes>.
        """
        treelets = treelet_funct(G)

        # dv is determined by the number of summands for FI + summands of FO

        if self.cuda_available:
            dv = torch.zeros(len(G.nodes)).cuda()
        else:
            dv = torch.zeros(len(G.nodes))

        # FI: Pass in Full Graph (\forall xv), [(xu, xv) for xu \in parents(xv)]
        in_batch = []
        in_indices = {}
        index = 0
        for xv_id, xv_obj in G.nodes.items():
            in_indices[xv_id] = []
            xv_dense = dense_nodes[xv_id]
            for xu_id in xv_obj.parents:
                in_indices[xv_id].append(index)
                index += 1

                xu_dense = dense_nodes[xu_id]
                in_batch.append(torch.cat([xu_dense, xv_dense], dim = 0))
                dv[xv_id] += 1

        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                in_sum = torch.zeros([len(G.nodes), 256]).cuda()
            else:
                in_sum = torch.zeros([len(G.nodes), 256])
            for xv_id in G.nodes.items():
                in_indices[xv_id].append(xv_id)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            in_sum = self.FI(in_batch) 


        # FO: Sum over all (xv, children of xv) 
        in_batch = []
        out_indices = {}
        index = 0
        for xv_id, xv_obj in G.nodes.items():
            out_indices[xv_id] = []
            xv_dense = dense_nodes[xv_id]
            for xu_id in xv_obj.children:
                out_indices[xv_id].append(index)
                index += 1

                xu_dense = dense_nodes[xu_id]
                in_batch.append(torch.cat([xv_dense, xu_dense], dim = 0))
                dv[xv_id] += 1

        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                out_sum = torch.zeros([len(G.nodes), 256]).cuda()
            else:
                out_sum = torch.zeros([len(G.nodes), 256])
            for xv_id in G.nodes.items():
                out_indices[xv_id].append(xv_id)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            out_sum = self.FO(in_batch)

        # print("FO Output ", out_sum)

        # Treelets!!
        if self.cuda_available:
            ev = torch.zeros(len(G.nodes)).cuda()
        else:
            ev = torch.zeros(len(G.nodes))


        # Left Treelet: (xv, xu, xw)
        in_batch = []
        left_indices = {}
        index = 0
        for xv_id in G.nodes.keys():
            left_indices[xv_id] = []
            for _, xu_id, xw_id in treelets[xv_id][0]:
                left_indices[xv_id].append(index)
                index += 1

                xv_dense = dense_nodes[xv_id]
                xu_dense = dense_nodes[xu_id]
                xw_dense = dense_nodes[xw_id]
                in_batch.append(torch.cat([xv_dense, xu_dense, xw_dense]))
                ev[xv_id] += 1

        # WHEN LEFT_TREELETS IS EMPTY
        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                left_sum = torch.zeros([len(G.nodes), 256]).cuda()
            else:
                left_sum = torch.zeros([len(G.nodes), 256])
            for xv_id in G.nodes.keys():
                left_indices[xv_id].append(xv_id)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            left_sum = self.FL(in_batch)

        # print("FL Output ", left_sum)

        # Head Treelet: xu, xv, xw
        in_batch = []
        head_indices = {}
        index = 0
        for xv_id in G.nodes.keys():
            head_indices[xv_id] = []
            for xu_id, _, xw_id in treelets[xv_id][1]:
                head_indices[xv_id].append(index)
                index += 1

                xu_dense = dense_nodes[xu_id]
                xv_dense = dense_nodes[xv_id]
                xw_dense = dense_nodes[xw_id]
                in_batch.append(torch.cat([xu_dense, xv_dense, xw_dense]))
                ev[xv_id] += 1

        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                head_sum = torch.zeros([len(G.nodes), 256]).cuda()
            else:
                head_sum = torch.zeros([len(G.nodes), 256])
            for xv_id in G.nodes.keys():
                head_indices[xv_id].append(xv_id)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            head_sum = self.FH(in_batch)


        # print("FH Output ", head_sum)

        # Right Treelet: xu, xw, xv
        in_batch = []
        right_indices = {}
        index = 0
        for xv_id in G.nodes.keys():
            right_indices[xv_id] = []
            for xu_id, xw_id, _ in treelets[xv_id][2]:
                right_indices[xv_id].append(index)
                index += 1

                xu_dense = dense_nodes[xu_id]
                xw_dense = dense_nodes[xw_id]
                xv_dense = dense_nodes[xv_id]
                in_batch.append(torch.cat([xu_dense, xw_dense, xv_dense]))
                ev[xv_id] += 1

        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            if self.cuda_available:
                right_sum = torch.zeros([len(G.nodes), 256]).cuda()
            else:
                right_sum = torch.zeros([len(G.nodes), 256])
            for xv_id in G.nodes.keys():
                right_indices[xv_id].append(xv_id)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            right_sum = self.FR(in_batch)

        # print("FR Output ", right_sum)


        # Add and then send to FP to update all the nodes!
        in_out_sum = []
        treelet_sum = []
        for xv_id in G.nodes.keys():
            new_in_sum = torch.sum(in_sum[in_indices[xv_id], :], dim = 0)
            new_out_sum = torch.sum(out_sum[out_indices[xv_id], :], dim = 0)

            new_left_sum = torch.sum(left_sum[left_indices[xv_id], :], dim = 0)
            new_head_sum = torch.sum(head_sum[head_indices[xv_id], :], dim = 0)
            new_right_sum = torch.sum(right_sum[right_indices[xv_id], :], dim = 0)

            # Append in_out_sum only if the number of summands is not zero. Else, append 0's.
            temp = new_in_sum + new_out_sum
            if len(temp.shape) != 0:
                if dv[xv_id] == 0: # xv has 0 degree
                    if self.cuda_available:
                        in_out_sum.append(torch.zeros(256).cuda())
                    else:
                        in_out_sum.append(torch.zeros(256))
                else:
                    in_out_sum.append(temp / dv[xv_id])
            else:
                if self.cuda_available:
                    in_out_sum.append(torch.zeros(256).cuda())
                else:
                    in_out_sum.append(torch.zeros(256))

            # Append treelet_sum only if the number of summands is not zero. Else, append 0's.
            temp = new_left_sum + new_head_sum + new_right_sum
            if len(temp.shape) != 0:
                if ev[xv_id] == 0:
                    if self.cuda_available:
                        treelet_sum.append(torch.zeros(256).cuda())
                    else:
                        treelet_sum.append(torch.zeros(256))
                else:
                    treelet_sum.append(temp / ev[xv_id])
            else:
                if self.cuda_available:
                    treelet_sum.append(torch.zeros(256).cuda())
                else:
                    treelet_sum.append(torch.zeros(256))

        in_out_sum = torch.stack(in_out_sum, dim = 0)
        treelet_sum = torch.stack(treelet_sum, dim = 0)
        # try:
        # except:
        #     print(treelet_sum)
        #     assert True is False

        # print("FP Inputs ", dense_nodes, in_out_sum, treelet_sum)

        # Now Add and Send All nodes to FP
        new_nodes = self.FP(dense_nodes + in_out_sum + treelet_sum)

        # print("FP Output ", new_nodes)

        return new_nodes

    def graph_to_one_hot(self, G):
        """
        Given a graph object, return an array of one-hot vectors.
        """
        NUM_TOKENS = 1909
        one_hot_graph = []
        for _, node in G.nodes.items():
            token = node.token
            token_index = self.token_to_index[token]
            one_hot_token = np.zeros(NUM_TOKENS)
            one_hot_token[token_index] = 1
            one_hot_graph.append(one_hot_token)
        one_hot_graph = np.array(one_hot_graph)
        return one_hot_graph


    def forward(self, conjecture_graphs, statement_graphs):
        """
        @ Args:
            conjecture_graph (arr-like of Graph Objects): Inter-graph Batch of Conjectures
            statement_graph (same)
        """
        conj_embedding_batch = []
        state_embedding_batch = []
        for i in range(len(conjecture_graphs)): # Iterate over inter-graph-batch.
            conjecture_graph = conjecture_graphs[i]
            statement_graph = statement_graphs[i]

            # Map graph object to an array of one hot vectors
            conj_one_hot = self.graph_to_one_hot(conjecture_graph)
            state_one_hot = self.graph_to_one_hot(statement_graph)

            # Map one_hot vectors of full graph into dense vectors of full graph
            if torch.cuda.is_available():
                conj_dense = torch.stack([self.dense_map(torch.Tensor(node).cuda()) for node in conj_one_hot])
                state_dense = torch.stack([self.dense_map(torch.Tensor(node).cuda()) for node in state_one_hot])
            else:
                conj_dense = torch.stack([self.dense_map(torch.Tensor(node)) for node in conj_one_hot])
                state_dense = torch.stack([self.dense_map(torch.Tensor(node)) for node in state_one_hot])


            # Iterate equations 1 or 2.
            for t in range(self.num_steps):
                # print("Conjecture Dense: ", conj_dense)
                # print("Statement Dense: ", state_dense)
                conj_dense = self.fullPass(conj_dense, conjecture_graph)
                state_dense = self.fullPass(state_dense, statement_graph)
                # print("Conjecture Dense: ", conj_dense)
                # print("Statement Dense: ", state_dense)
                

            # Finished Updating. max-pool over all nodes in the graph
            conj_embedding = self.max_pool_dense_graph(conj_dense)
            state_embedding = self.max_pool_dense_graph(state_dense)
            # print(conj_embedding)
            # print(state_embedding)

            conj_embedding_batch.append(conj_embedding)
            state_embedding_batch.append(state_embedding)

        conj_batch = torch.stack(conj_embedding_batch, dim = 0)
        state_batch = torch.stack(state_embedding_batch, dim = 0)

        # Classify
        prediction = self.Classifier(conj_batch, state_batch)

        return prediction



# if __name__ == "__main__":
    # graph_to_index_offline()

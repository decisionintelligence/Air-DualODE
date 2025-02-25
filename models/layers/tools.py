class Air_Attrs:
    def __init__(self, args):
        self.args = args
        self.adj_mx = args.adj_mx
        self.num_nodes = args.adj_mx.shape[0]
        self.num_edges = args.edge_index.shape[1]

        self.seq_len = int(args.model.seq_len)
        self.horizon = int(args.model.horizon)
        self.input_dim = int(args.model.input_dim)
        self.X_dim = int(args.model.X_dim)
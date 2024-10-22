# from dataset/params from main.py, train TSCP2/TNC/TS2vec model.

from data import *
from model import *
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

class ReprModel:
    def __init__(self, args) -> None:
        self.args = args
    def fit(self):
        repr_name = self.args.repr
        device = "cuda:" + self.args.gpu
        data_name = self.args.data
        repr_dims = self.args.dim
        input_dims = self.args.input_dim
        window_size = self.args.window
        slide_size = self.args.slide
        batch_size = self.args.batch
        epochs = self.args.epoch
        LR=self.args.lr
        depth=self.args.depth
        nnum = self.args.nnum
        nrange = self.args.nrange
        train_data = np.load(f"./datasets/{self.args.data}_X_long.npy")
        print(f"{self.args.data} {train_data.shape} imported")

        if repr_name == "TSCP2":
            model = TSCP2(
                input_dims=input_dims,
                output_dims=repr_dims,
                window_size=window_size,
                depth=depth
                ).to(device)
            train_dataset = TSCP2DS(
                torch.Tensor(train_data), 
                window=window_size, 
                slide=slide_size
            )

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            model.train()
            size = len(train_dataloader.dataset)*window_size
            for i in trange(epochs):
                loader_bar = tqdm(train_dataloader, leave=False)
                for batch, (X1, X2) in enumerate(loader_bar):
                    # print(X1.shape, X2.shape)
                    X1, X2 = X1.to(device), X2.to(device)

                    # Compute prediction error
                    repr1 = model(X1)[:,None,:]
                    repr2 = model(X2)[:,None,:]
                    loss = instance_contrastive_loss(repr1, repr2)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loader_bar.set_description(
                        f"{data_name=} {repr_dims=} {window_size=} {slide_size=} {batch_size=} {epochs=} {LR=} train loss={loss.item():.3f}"
                    )
                print(loss.item())

            
            train_data_torch = torch.Tensor(train_data).to(device)
            train_data_pad = F.pad(train_data_torch.transpose(0,1), (window_size//2-1, window_size//2), mode="replicate")
            train_data_pad = train_data_pad.transpose(0,1)

            test_dataset = WindowedTS_NoLabel(train_data_pad, window=window_size, slide=1)
            test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)


            model.eval()
            test_repr_long = []
            with torch.no_grad():
                for X in tqdm(test_dataloader):
                    X = X.to(device)
                    test_repr_long.append(model(X).cpu().numpy())
            test_repr_long = np.concatenate(test_repr_long)
        
        
        elif repr_name == "TNC":
            model = TNC(input_dims=input_dims, output_dims=repr_dims, window_size=window_size).to(device)
            train_dataset = TNCDS(
                data=train_data, 
                window=window_size,
                n_num=nnum,
                n_range=nrange
                )

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            model.train()
            size = len(train_dataloader.dataset)*window_size
            loss_fn = torch.nn.BCEWithLogitsLoss()
            print(f"{epochs=},{len(train_dataloader)=},{len(train_dataset)=}")
            for i in trange(epochs):
                loader_bar = tqdm(train_dataloader, leave=False)
                for anchor, pos, neg in loader_bar:
                    anchor, pos, neg = torch.Tensor(anchor).to(device), torch.Tensor(pos).to(device), torch.Tensor(neg).to(device)
                    # x: batch, window, input_dim
                    # pos: batch, nnum, window, input_dim
                    # neg: batch, nnum, window, input_dim
                    pos = torch.reshape(pos,(len(anchor)*nnum,window_size,input_dims)) # batch*nnum, window, input_dim shape // '[1280, 50, 6]' is invalid for input of size 120000
                    neg = torch.reshape(neg,(len(anchor)*nnum,window_size,input_dims)) # batch*nnum, window, input_dim 

                    anchor = model(anchor)
                    anchor = torch.repeat_interleave(anchor, nnum, dim=0) # batch*nnum, window, input_dim
                    pos = model(pos)
                    neg = model(neg)
                    # generate labels for pos/neg
                    pos_output = model.disc(anchor, pos)
                    neg_output = model.disc(anchor, neg)

                    pos_y = torch.ones((len(pos_output),1)).to(device)
                    neg_y = torch.zeros((len(neg_output),1)).to(device)

                    loss = loss_fn(pos_output, pos_y) + loss_fn(neg_output, neg_y) 

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loader_bar.set_description(
                        f"{data_name=} {repr_dims=} {window_size=} {slide_size=} {batch_size=} {epochs=} {LR=} train loss={loss.item():.3f}"
                    )
                    
                print(loss.item())


            train_data_torch = torch.Tensor(train_data).to(device)
            train_data_pad = F.pad(train_data_torch.transpose(0,1), (window_size//2-1, window_size//2), mode="replicate")
            train_data_pad = train_data_pad.transpose(0,1)

            test_dataset = WindowedTS_NoLabel(train_data_pad, window=window_size, slide=1)
            test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)


            num_batches = len(test_dataloader)
            model.eval()
            size = len(test_dataloader.dataset)
            test_repr_long = []
            with torch.no_grad():
                for X in tqdm(test_dataloader):
                    X = X.to(device)
                    test_repr_long.append(model(X).cpu().numpy())
            test_repr_long = np.concatenate(test_repr_long)
        else:
            raise NotImplementedError
        return test_repr_long


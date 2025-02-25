from Data_Provider.data_loader import Dataset_Beijing1718, Dataset_KnowAir
from torch.utils.data import DataLoader

data_dict = {
    'Beijing1718': Dataset_Beijing1718,
    'KnowAir': Dataset_KnowAir
}


def data_provider(args, flag):
    data_args = args.data
    model_args = args.model
    Data = data_dict[data_args.data_name]

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    else:
        shuffle_flag = False
        drop_last = False
    batch_size = data_args.batch_size

    if data_args.data_name == "Beijing1718_old":
        data_set = Data(
            root_path=data_args.root_path,
            flag=flag
        )
    else:
        data_set = Data(
            root_path=data_args.root_path,
            flag=flag,
            seq_len=model_args.seq_len,
            pred_len=model_args.horizon,
            freq=data_args.interval,
            embed=data_args.embed,
            scale=True,
            normalized_col=data_args.normalized_columns
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
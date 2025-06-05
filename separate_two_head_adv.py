import argparse
import os
import time

import torch

from beer import BeerData, BeerAnnotation,Beer_correlated
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import RNP_0908,Adv_generator
from train_util import train_separate_two_head,validate_separate_acc_on_valset
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser(
        description="SR")
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')


    # model parameters
    parser.add_argument('--correlated',
                        type=int,
                        default=0,
                        help='1:spurious correlation, 0: no spurious')
    parser.add_argument('--fr',
                        type=int,
                        default=0,
                        help='1:share gen and pred, 0:')
    parser.add_argument('--adv_sparse',
                        type=int,
                        default=1,
                        help='add sparsity to adv head')
    parser.add_argument('--adv_continue',
                        type=int,
                        default=1,
                        help='add sparsity to adv head')
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epoch')
    parser.add_argument('--use_hinge',
                        type=int,
                        default=1,
                        help='1:use hinge loss for the adv attack. 0: use 0.5 as the oppisite labels')
    parser.add_argument('--open_attack',
                        type=int,
                        default=1,
                        help='1:use hinge loss for the adv attack. 0: use 0.5 as the oppisite labels')
    parser.add_argument('--open_epoch',
                        type=int,
                        default=0,
                        help='start to attack')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--adv_lr_rate',
                        type=float,
                        default=1,
                        help='the lr of the adv head')
    parser.add_argument('--pred_lr_rate',
                        type=float,
                        default=1,
                        help='the lr of the adv head')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument('--adv_lambda',
                        type=float,
                        default=1.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='1',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################

if args.data_type=='beer':      
    if args.correlated==0:
        print('decorrelated')
        train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
    else:
        print('correlated')
        train_data = Beer_correlated(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = Beer_correlated(args.data_dir, args.aspect, 'dev', word2idx,balance=True)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':     
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
rnp_model=RNP_0908(args)
adv_gen=Adv_generator(args)
rnp_model.to(device)
adv_gen.to(device)

######################
# Training
######################
lr_gen = args.lr
lr_gen_adv = args.lr * args.adv_lr_rate
lr_pred = args.lr / args.pred_lr_rate
if args.fr==0:
    gen_para=[]
    pred_para=[]
    for module in rnp_model.gen_para:
        for name, p in module.named_parameters():
            if p.requires_grad==True:
                gen_para.append(p)

    for module in rnp_model.pred_para:
        for name, p in module.named_parameters():
            if p.requires_grad==True:
                pred_para.append(p)
    para=[
        {'params': gen_para, 'lr':lr_gen},
    {'params': pred_para, 'lr':lr_pred}
    ]
    optimizer = torch.optim.Adam(para)
    optimizer_adv=torch.optim.Adam(adv_gen.parameters(),lr=lr_gen_adv)
else:
    optimizer = torch.optim.Adam(rnp_model.parameters(),lr=lr_gen)
    optimizer_adv = torch.optim.Adam(adv_gen.parameters(), lr=lr_gen_adv)

print('lr_gen={},lr_gen_adv={},lr_pred={}'.format(lr_gen,lr_gen_adv,lr_pred))


######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]
for epoch in range(args.epochs):
    if args.open_epoch>epoch:
        args.open_attack=0

    else:
        args.open_attack = 1



    start = time.time()
    rnp_model.train()
    adv_gen.train()
    normal_acc_set, adv_acc_set = train_separate_two_head(rnp_model,adv_gen, optimizer,optimizer_adv, train_loader, device, args,(writer,epoch))
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print('-------')
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, normal_acc_set[1],
                                                                                                   normal_acc_set[0], normal_acc_set[2],
                                                                                                   normal_acc_set[3]))
    print("adv_head: recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(
                                                                                                   adv_acc_set[1],
                                                                                                   adv_acc_set[0],
                                                                                                   adv_acc_set[2],
                                                                                                   adv_acc_set[3]))
    print('-------')

    writer.add_scalar('train_acc',normal_acc_set[3],epoch)
    writer.add_scalar('train_acc_adv', adv_acc_set[3], epoch)
    writer.add_scalar('time',time.time()-strat_time,epoch)

    print("Validate")

    normal_acc_set, adv_acc_set = validate_separate_acc_on_valset(rnp_model,adv_gen, dev_loader, device)

    print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch,
                                                                                                   normal_acc_set[1],
                                                                                                   normal_acc_set[0],
                                                                                                   normal_acc_set[2],
                                                                                                   normal_acc_set[3]))
    print("adv_head: recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(
        adv_acc_set[1],
        adv_acc_set[0],
        adv_acc_set[2],
        adv_acc_set[3]))
    print("Validate Sentence")
    validate_dev_sentence(rnp_model, dev_loader, device, (writer, epoch))

    print('-------')

    writer.add_scalar('dev_acc',normal_acc_set[3],epoch)
    writer.add_scalar('dev_acc_adv', adv_acc_set[3], epoch)


    print("Annotation")
    annotation_results = validate_share(rnp_model, annotation_loader, device)
    print(
        "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
        % (100 * annotation_results[0], 100 * annotation_results[1],
           100 * annotation_results[2], 100 * annotation_results[3]))
    writer.add_scalar('f1',100 * annotation_results[3],epoch)
    writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
    writer.add_scalar('p', 100 * annotation_results[1], epoch)
    writer.add_scalar('r', 100 * annotation_results[2], epoch)
    print("Annotation Sentence")
    validate_annotation_sentence(rnp_model, annotation_loader, device)
    print("Rationale")
    validate_rationales(rnp_model, annotation_loader, device,(writer,epoch))
    if normal_acc_set[3]>acc_best_dev[-1]:
        acc_best_dev.append(normal_acc_set[3])
        best_dev_epoch.append(epoch)
        f1_best_dev.append(annotation_results[3])
    if best_all<annotation_results[3]:
        best_all=annotation_results[3]
    print('---end epoch----')
    print('')


print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)
if args.save==1:
    if args.data_type=='beer':
        if args.fr==0:
            torch.save(rnp_model.state_dict(),'./trained_model/beer/aspect{}_rnp_sp{}_adv{}.pkl'.format(args.aspect,args.sparsity_percentage,args.adv_lambda))
            torch.save(adv_gen.state_dict(),'./trained_model/beer/aspect{}_rnpadv_sp{}_adv{}.pkl'.format(args.aspect,args.sparsity_percentage,args.adv_lambda))
        else:
            torch.save(rnp_model.state_dict(),
                       './trained_model/beer/aspect{}_fr_sp{}_adv{}.pkl'.format(args.aspect, args.sparsity_percentage,args.adv_lambda))
            torch.save(adv_gen.state_dict(),
                       './trained_model/beer/aspect{}_fradv_sp{}_adv{}.pkl'.format(args.aspect, args.sparsity_percentage,args.adv_lambda))
        print('save the model')
    elif args.data_type=='hotel':
        torch.save(rnp_model.state_dict(), './trained_model/hotel/aspect{}_dis{}.pkl'.format(args.aspect, args.dis_lr))
        print('save the model')
else:
    print('not save')
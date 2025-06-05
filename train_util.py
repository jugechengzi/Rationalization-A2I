import torch
import torch.nn.functional as F

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np
import math



def train_noshare(model, optimizer, dataset, device, args,writer_epoch,grad):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        rationales, logits = model(inputs, masks)

        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        l_logits=torch.mean(logits)
        l_logits.backward(retain_graph=True)
        for k,v in model.gen.named_parameters():
            if k == "weight_ih_l0":
                g=abs(v.grad.clone().detach())
                grad.append(g)
        optimizer.zero_grad()
        improve=torch.mean((grad[-1]-grad[0])/grad[0])
        writer_epoch[0].add_scalar('grad', improve, writer_epoch[1]*len(dataset)+batch)

        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_sp_norm(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        rationales, logits = model(inputs, masks)

        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr

        loss.backward()
        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, f1_score, accuracy

def train_rnp_noacc(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        rationales=model.get_rationale(inputs,masks)
        logits=model.pred_with_rationale(inputs,masks,torch.detach(rationales))

        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = sparsity_loss+cls_loss


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, f1_score, accuracy

def train_rnp_noacc_to_classifier(model,classifier, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    TP_c = 0
    TN_c = 0
    FN_c = 0
    FP_c = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len = len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        rationales=model.get_rationale(inputs,masks)
        logits=model.pred_with_rationale(inputs,masks,torch.detach(rationales))

        classifier_logits=classifier(inputs,masks,torch.detach(rationales[:,:,1]))

        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = sparsity_loss+cls_loss


        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        cls_soft_logits_classifier = torch.softmax(classifier_logits, dim=-1)

        _, pred = torch.max(cls_soft_logits, dim=-1)
        _, pred_classifier = torch.max(cls_soft_logits_classifier, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()


        TP_c += ((pred_classifier == 1) & (labels == 1)).cpu().sum()
        TN_c += ((pred_classifier == 0) & (labels == 0)).cpu().sum()
        FN_c += ((pred_classifier == 0) & (labels == 1)).cpu().sum()
        FP_c += ((pred_classifier == 1) & (labels == 0)).cpu().sum()


        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision_c = TP_c / (TP_c + FP_c)
    recall_c = TP_c / (TP_c + FN_c)
    f1_score_c = 2 * recall_c * precision_c / (recall_c + precision_c)
    accuracy_c = (TP_c + TN_c) / (TP_c + TN_c + FP_c + FN_c)

    return (precision, recall, f1_score, accuracy),(precision_c, recall_c, f1_score_c, accuracy_c)

def classfy(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        logits = model(inputs, masks)

        cls_loss =F.cross_entropy(logits, labels)


        loss = 0.9*cls_loss

        loss.backward()

        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy



def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def get_grad(model,dataloader,p,use_rat,device):            
    data=0
    model.train()
    grad=[]
    for batch,d in enumerate(dataloader):
        data=d
        inputs, masks, labels = data
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationale,logit,embedding2,cls_embed=model.grad(inputs, masks)
        loss=torch.mean(torch.softmax(logit,dim=-1)[:,1])
        cls_embed.retain_grad()
        loss.backward()
        if use_rat==0:
            k_mask=masks
        elif use_rat==1:
            k_mask=rationale[:,:,1]
        masked_grad=cls_embed.grad*k_mask.unsqueeze(-1)
        gradtemp=torch.sum(abs(masked_grad),dim=1)       
        gradtemp=gradtemp/torch.sum(k_mask,dim=-1).unsqueeze(-1)      
        gradtempmask = gradtemp
        norm_grad=torch.linalg.norm(gradtempmask, ord=p, dim=1)           
        grad.append(norm_grad.clone().detach().cpu())
    grad=torch.cat(grad,dim=0)
    tem=[]
    for g in grad:
        if math.isnan(g.item()):
            continue
        else:
            tem.append(g)

    tem=torch.tensor(tem)
    maxg=torch.max(tem)*1000
    meang=torch.mean(tem)*1000
    return maxg,meang


def train_two_head(model, optimizer, opt_adv, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0


    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    train_sp_adv=[]
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

      
        name_adv=[]
        for idx,p in model.gen_fc_adv.named_parameters():
            if p.requires_grad==True:
                name_adv.append(idx)
                p.requires_grad=False

        optimizer.zero_grad()
        opt_adv.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        labels_adv=(torch.ones_like(labels)-labels).to(device)


        rationales, rationales_adv = model.get_rationale(inputs, masks)

        pred_logits=model.pred_with_rationale(inputs, masks, rationales)

        pred_logits_adv=model.pred_with_rationale(inputs, masks, rationales_adv)

        cls_loss = args.cls_lambda * F.cross_entropy(pred_logits, labels)

        cross_entropy_adv= F.cross_entropy(pred_logits_adv, labels_adv,reduction='none')    
        temp=torch.ones_like(pred_logits_adv)*0.5   
        temp1=F.cross_entropy(temp, labels_adv,reduction='none')
        temp2=torch.stack([cross_entropy_adv,temp1],dim=0)
        cls_loss_adv=args.cls_lambda * torch.mean(torch.min(temp2,dim=0)[0])

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss - args.adv_lambda*cls_loss_adv
        loss.backward()
        optimizer.step()


        tp,tn,fn,fp=calculate_acc(pred_logits,labels)

        TP += tp
        TN += tn
        FN += fn
        FP += fp


        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

        optimizer.zero_grad()
        opt_adv.zero_grad()


        for idx, p in model.gen_fc_adv.named_parameters():
            if idx in name_adv:
                p.requires_grad = True


        name_gen_pred=[]
        for i in model.gen_pred:
            for name, p in i.named_parameters():
                if p.requires_grad == True:
                    name_gen_pred.append(name)
                    p.requires_grad = False


        rationales, rationales_adv = model.get_rationale(inputs, masks)
        pred_logits_adv = model.pred_with_rationale(inputs, masks, rationales_adv)
        cross_entropy_adv = args.cls_lambda*F.cross_entropy(pred_logits_adv, labels_adv)

        sparsity_loss_adv = args.sparsity_lambda * get_sparsity_loss(
            rationales_adv[:, :, 1], masks, args.sparsity_percentage)


        train_sp_adv.append(
            (torch.sum(rationales_adv[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss_adv = args.continuity_lambda * get_continuity_loss(
            rationales_adv[:, :, 1])

        if args.adv_sparse==1:
            loss = cross_entropy_adv + sparsity_loss_adv + continuity_loss_adv
        else:
            loss = cross_entropy_adv


        loss.backward()
        opt_adv.step()

        tp_adv,tn_adv,fn_adv,fp_adv=calculate_acc(pred_logits_adv,labels_adv)


        TP_adv += tp_adv
        TN_adv += tn_adv
        FN_adv += fn_adv
        FP_adv += fp_adv

        for i in model.gen_pred:
            for name, p in i.named_parameters():
                if name in name_gen_pred:
                    p.requires_grad=True


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision_adv = TP_adv / (TP_adv + FP_adv)
    recall_adv = TP_adv / (TP_adv + FN_adv)
    f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
    accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)



    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])

    return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)


def train_separate_two_head(rnp_model,adv_gen, optimizer, opt_adv, dataset, device, args,writer_epoch):

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0


    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    train_sp_adv=[]
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        name_adv=[]
        for idx,p in adv_gen.named_parameters():
            if p.requires_grad==True:
                name_adv.append(idx)
                p.requires_grad=False

        optimizer.zero_grad()
        opt_adv.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        labels_adv=(torch.ones_like(labels)-labels).to(device)


        rationales= rnp_model.get_rationale(inputs, masks)
        rationales_adv=adv_gen.get_rationale(inputs, masks)

        pred_logits=rnp_model.pred_with_rationale(inputs, masks, rationales)

        pred_logits_adv=rnp_model.pred_with_rationale(inputs, masks, rationales_adv)

        cls_loss = args.cls_lambda * F.cross_entropy(pred_logits, labels)

        if args.use_hinge==1:
            cross_entropy_adv= F.cross_entropy(pred_logits_adv, labels_adv,reduction='none')    
            temp=torch.ones_like(pred_logits_adv)*0.5   
            temp1=F.cross_entropy(temp, labels_adv,reduction='none')
            temp2=torch.stack([cross_entropy_adv,temp1],dim=0)
            cls_loss_adv=args.cls_lambda * torch.mean(torch.min(temp2,dim=0)[0])

        else:
            half_lables=torch.ones_like(pred_logits_adv)*0.5
            cross_entropy_adv = F.cross_entropy(pred_logits_adv, half_lables)
            cls_loss_adv=args.cls_lambda * cross_entropy_adv


        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        if args.use_hinge==1:
            loss = cls_loss + sparsity_loss + continuity_loss - args.open_attack*args.adv_lambda*cls_loss_adv
        else:
            loss = cls_loss + sparsity_loss + continuity_loss + args.open_attack*args.adv_lambda * cls_loss_adv
        loss.backward()
        optimizer.step()


        tp,tn,fn,fp=calculate_acc(pred_logits,labels)


        TP += tp
        TN += tn
        FN += fn
        FP += fp


        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()


        optimizer.zero_grad()
        opt_adv.zero_grad()


        for idx, p in adv_gen.named_parameters():
            if idx in name_adv:
                p.requires_grad = True

        name_gen_pred=[]

        for name, p in rnp_model.named_parameters():
            if p.requires_grad == True:
                name_gen_pred.append(name)
                p.requires_grad = False


        rationales_adv = adv_gen.get_rationale(inputs, masks)
        pred_logits_adv = rnp_model.pred_with_rationale(inputs, masks, rationales_adv)
        cross_entropy_adv = args.cls_lambda*F.cross_entropy(pred_logits_adv, labels_adv)

        sparsity_loss_adv = args.sparsity_lambda * get_sparsity_loss(
            rationales_adv[:, :, 1], masks, args.sparsity_percentage)


        train_sp_adv.append(
            (torch.sum(rationales_adv[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss_adv = args.continuity_lambda * get_continuity_loss(
            rationales_adv[:, :, 1])


        loss = cross_entropy_adv + args.adv_sparse*sparsity_loss_adv + args.adv_continue*continuity_loss_adv


        loss.backward()
        opt_adv.step()

        tp_adv,tn_adv,fn_adv,fp_adv=calculate_acc(pred_logits_adv,labels_adv)


        TP_adv += tp_adv
        TN_adv += tn_adv
        FN_adv += fn_adv
        FP_adv += fp_adv


        for name, p in rnp_model.named_parameters():
            if name in name_gen_pred:
                p.requires_grad=True


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision_adv = TP_adv / (TP_adv + FP_adv)
    recall_adv = TP_adv / (TP_adv + FN_adv)
    f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
    accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)



    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp_adv', np.mean(train_sp_adv), writer_epoch[1])
    print('train adv sparsity={}'.format(np.mean(train_sp_adv)))

    return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)

def train_postattack(rnp_model,adv_gen, optimizer, opt_adv, dataset, device, args,writer_epoch):

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0


    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    train_sp_adv=[]
    batch_len=len(dataset)


    name_gen_pred = []

    for name, p in rnp_model.named_parameters():
        if p.requires_grad == True:
            name_gen_pred.append(name)
            p.requires_grad = False

    for (batch, (inputs, masks, labels)) in enumerate(dataset):



        optimizer.zero_grad()
        opt_adv.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        labels_adv=(torch.ones_like(labels)-labels).to(device)


        optimizer.zero_grad()
        opt_adv.zero_grad()


        rationales_adv = adv_gen.get_rationale(inputs, masks)
        pred_logits_adv = rnp_model.pred_with_rationale(inputs, masks, rationales_adv)
        cross_entropy_adv = args.cls_lambda*F.cross_entropy(pred_logits_adv, labels_adv)

        sparsity_loss_adv = args.sparsity_lambda * get_sparsity_loss(
            rationales_adv[:, :, 1], masks, args.sparsity_percentage)


        train_sp_adv.append(
            (torch.sum(rationales_adv[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss_adv = args.continuity_lambda * get_continuity_loss(
            rationales_adv[:, :, 1])

        loss = cross_entropy_adv + args.adv_sparse*sparsity_loss_adv + args.adv_continue*continuity_loss_adv


        loss.backward()
        opt_adv.step()

        tp_adv,tn_adv,fn_adv,fp_adv=calculate_acc(pred_logits_adv,labels_adv)



        TP_adv += tp_adv
        TN_adv += tn_adv
        FN_adv += fn_adv
        FP_adv += fp_adv


        for name, p in rnp_model.named_parameters():
            if name in name_gen_pred:
                p.requires_grad=True


    precision = 0
    recall = 0
    f1_score = 0
    accuracy = 0

    precision_adv = TP_adv / (TP_adv + FP_adv)
    recall_adv = TP_adv / (TP_adv + FN_adv)
    f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
    accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)

    writer_epoch[0].add_scalar('train_sp_adv', np.mean(train_sp_adv), writer_epoch[1])
    print('train adv sparsity={}'.format(np.mean(train_sp_adv)))

    return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)

def train_separate_two_head_normal_generator(rnp_model,adv_gen, opt_pred,opt_gen, opt_adv, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0


    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    train_sp_adv=[]
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        opt_pred.zero_grad()
        opt_gen.zero_grad()
        opt_adv.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        labels_adv=(torch.ones_like(labels)-labels).to(device)

        rationales= rnp_model.get_rationale(inputs, masks)

        pred_logits=rnp_model.pred_with_rationale(inputs, masks, rationales)

        cls_loss = args.cls_lambda * F.cross_entropy(pred_logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        loss.backward()
        opt_pred.step()
        opt_gen.step()

        tp,tn,fn,fp=calculate_acc(pred_logits,labels)

        TP += tp
        TN += tn
        FN += fn
        FP += fp

        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()


        opt_gen.zero_grad()
        opt_pred.zero_grad()
        opt_adv.zero_grad()


        name_gen_pred=[]

        for name, p in rnp_model.named_parameters():
            if p.requires_grad == True:
                name_gen_pred.append(name)
                p.requires_grad = False


        rationales_adv = adv_gen.get_rationale(inputs, masks)
        pred_logits_adv = rnp_model.pred_with_rationale(inputs, masks, rationales_adv)
        cross_entropy_adv = args.cls_lambda*F.cross_entropy(pred_logits_adv, labels_adv)

        sparsity_loss_adv = args.sparsity_lambda * get_sparsity_loss(
            rationales_adv[:, :, 1], masks, args.sparsity_percentage)


        train_sp_adv.append(
            (torch.sum(rationales_adv[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss_adv = args.continuity_lambda * get_continuity_loss(
            rationales_adv[:, :, 1])

        if args.adv_sparse==1:
            loss = cross_entropy_adv + sparsity_loss_adv + continuity_loss_adv
        else:
            loss = cross_entropy_adv


        loss.backward()
        opt_adv.step()

        tp_adv,tn_adv,fn_adv,fp_adv=calculate_acc(pred_logits_adv,labels_adv)

        TP_adv += tp_adv
        TN_adv += tn_adv
        FN_adv += fn_adv
        FP_adv += fp_adv


        for name, p in rnp_model.named_parameters():
            if name in name_gen_pred:
                p.requires_grad=True


    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision_adv = TP_adv / (TP_adv + FP_adv)
    recall_adv = TP_adv / (TP_adv + FN_adv)
    f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
    accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)



    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp_adv', np.mean(train_sp_adv), writer_epoch[1])
    print('train adv sparsity={}'.format(np.mean(train_sp_adv)))

    return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)

def train_separate_two_head_attacker_predictor(rnp_model,adv_gen, opt_pred,opt_gen, opt_adv, dataset, device, args,writer_epoch):


    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0


    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    train_sp_adv=[]
    batch_len=len(dataset)


    name_adv = []
    for idx, p in adv_gen.named_parameters():
        if p.requires_grad == True:
            name_adv.append(idx)
            p.requires_grad = False

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        opt_pred.zero_grad()
        opt_gen.zero_grad()
        opt_adv.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        labels_adv=(torch.ones_like(labels)-labels).to(device)

        rationales_adv=adv_gen.get_rationale(inputs, masks)

        pred_logits_adv=rnp_model.pred_with_rationale(inputs, masks, rationales_adv)

        if args.use_hinge==1:
            cross_entropy_adv= F.cross_entropy(pred_logits_adv, labels_adv,reduction='none')  
            temp1=F.cross_entropy(temp, labels_adv,reduction='none')
            temp2=torch.stack([cross_entropy_adv,temp1],dim=0)
            cls_loss_adv=args.cls_lambda * torch.mean(torch.min(temp2,dim=0)[0])

        else:
            half_lables=torch.ones_like(pred_logits_adv)*0.5
            cross_entropy_adv = F.cross_entropy(pred_logits_adv, half_lables)
            cls_loss_adv=args.cls_lambda * cross_entropy_adv


        if args.use_hinge==1:
            loss = - args.adv_lambda*cls_loss_adv
        else:
            loss = args.adv_lambda * cls_loss_adv
        loss.backward()

        opt_pred.step()
        tp_adv,tn_adv,fn_adv,fp_adv=calculate_acc(pred_logits_adv,labels_adv)

        TP_adv += tp_adv
        TN_adv += tn_adv
        FN_adv += fn_adv
        FP_adv += fp_adv

        pred_para = []
        for module in rnp_model.pred_para:
            for idx, p in module.named_parameters():
                if p.requires_grad == True:
                    pred_para.append(idx)
                    p.requires_grad=False

        opt_pred.zero_grad()
        opt_gen.zero_grad()
        opt_adv.zero_grad()

        rationales = rnp_model.get_rationale(inputs, masks)

        pred_logits = rnp_model.pred_with_rationale(inputs, masks, rationales)

        cls_loss = args.cls_lambda * F.cross_entropy(pred_logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        loss.backward()
        opt_gen.step()
        opt_gen.zero_grad()

        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

        tp, tn, fn, fp = calculate_acc(pred_logits, labels)

        TP += tp
        TN += tn
        FN += fn
        FP += fp

        for module in rnp_model.pred_para:
            for idx, p in module.named_parameters():
                if idx in pred_para:
                    p.requires_grad = True


    precision_adv = TP_adv / (TP_adv + FP_adv)
    recall_adv = TP_adv / (TP_adv + FN_adv)
    f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
    accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    writer_epoch[0].add_scalar('cls', cls_l, writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])


    for idx, p in adv_gen.named_parameters():
        if idx in name_adv:
            p.requires_grad = True

    return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)

def calculate_acc(pred_logits,labels):
    TP=0
    TN=0
    FN=0
    FP=0
    cls_soft_logits = torch.softmax(pred_logits, dim=-1)
    _, pred = torch.max(cls_soft_logits, dim=-1)

    TP += ((pred == 1) & (labels == 1)).cpu().sum()
    TN += ((pred == 0) & (labels == 0)).cpu().sum()
    FN += ((pred == 0) & (labels == 1)).cpu().sum()
    FP += ((pred == 1) & (labels == 0)).cpu().sum()

    return TP,TN,FN,FP


def validate_acc_on_valset(model, dev_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0

    model.eval()
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):

            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            labels_adv = (torch.ones_like(labels) - labels).to(device)

            rationales, rationales_adv = model.get_rationale(inputs, masks)

            pred_logits = model.pred_with_rationale(inputs, masks, rationales)

            pred_logits_adv = model.pred_with_rationale(inputs, masks, rationales_adv)

            tp, tn, fn, fp = calculate_acc(pred_logits, labels)
            TP += tp
            TN += tn
            FN += fn
            FP += fp

            tp_adv, tn_adv, fn_adv, fp_adv = calculate_acc(pred_logits_adv, labels_adv)
            TP_adv += tp_adv
            TN_adv += tn_adv
            FN_adv += fn_adv
            FP_adv += fp_adv

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision_adv = TP_adv / (TP_adv + FP_adv)
        recall_adv = TP_adv / (TP_adv + FN_adv)
        f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
        accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)

        return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)


def validate_separate_acc_on_valset(rnp_model,adv_gen, dev_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    TP_adv = 0
    TN_adv = 0
    FN_adv = 0
    FP_adv = 0

    rnp_model.eval()
    adv_gen.eval()
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):

            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            labels_adv = (torch.ones_like(labels) - labels).to(device)

            rationales= rnp_model.get_rationale(inputs, masks)
            rationales_adv=adv_gen.get_rationale(inputs, masks)

            pred_logits = rnp_model.pred_with_rationale(inputs, masks, rationales)

            pred_logits_adv = rnp_model.pred_with_rationale(inputs, masks, rationales_adv)

            tp, tn, fn, fp = calculate_acc(pred_logits, labels)
            TP += tp
            TN += tn
            FN += fn
            FP += fp

            tp_adv, tn_adv, fn_adv, fp_adv = calculate_acc(pred_logits_adv, labels_adv)
            TP_adv += tp_adv
            TN_adv += tn_adv
            FN_adv += fn_adv
            FP_adv += fp_adv

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision_adv = TP_adv / (TP_adv + FP_adv)
        recall_adv = TP_adv / (TP_adv + FN_adv)
        f1_score_adv = 2 * recall_adv * precision_adv / (recall_adv + precision_adv)
        accuracy_adv = (TP_adv + TN_adv) / (TP_adv + TN_adv + FP_adv + FN_adv)

        return (precision, recall, f1_score, accuracy), (precision_adv, recall_adv, f1_score_adv, accuracy_adv)




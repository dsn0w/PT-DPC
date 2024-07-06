import os
import sys
# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(abspath('/root/autodl-tmp/dsn/old/ijcai/FI_dynamic_nordorp_new.py'))))

import CLIP.clip as clip
import torch
import random
from PIL import ImageFile, Image
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from CLIP.clip.model import build_model
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True
savemodel = False
# savemodel = True
onlytest = False
# onlytest = True
loadbestmodel = False
# loadbestmodel = True
softchoice = True
# softchoice = False
# template = 'label'
template = 'sentence'
initT = 'a photo of '
# initT = 'This is a photo of '
# initT = 'a photo seems to express a feeling of '
# initT = 'an image to express a feeling like '
# initT = ' '
# initT = 'a photo seems to express a emotion of '
# initT = 'a picture seems to express some feelings like '
# initT = 'a photo seems to express some feelings like '
print(initT)
catedir = {}
catedirnlp = {}
catepath = '/data/dataset/cate info/FI/'
# catepath = '/root/autodl-tmp/dataset/cate info/FI/'
for file in os.listdir(catepath):
    with open(catepath+file, 'r') as catefile:
        cateinfo = catefile.readline().strip(' ')
        idx = '_'+file.strip('.txt')+'.jpg'
        catedir[file] = cateinfo
        wordlist = cateinfo.split(' ')
        wordinfo = ''
        for order in range(len(wordlist)):
            if order == len(wordlist) - 1:
                wordinfo = wordinfo + 'and ' + wordlist[order]
            else:
                wordinfo = wordinfo + wordlist[order] + ', '
        catedirnlp[file] = wordinfo

data_dir = '/data/dataset/FI/'
# data_dir = '/root/autodl-tmp/dataset/FI/'
classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
num_classes = len(classtype)
input_size = 224
batch_size = 64
seed = 1

num_epochs = 25
lr_FC = 0.01
mm_FC = 0.9
stepsize_FC = 6
gamma_FC = 0.1

# lr_BB = 1e-3
# wd_BB = 1e-2
# stepsize_BB = 4
# gamma_BB = 0.5
# warmupNum = 30
# warmupNum = 3


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    os.environ['PYTHONHASHSEED'] = str(seed)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Loaddata(torch.utils.data.Dataset):
    def __init__(self, setname):
        self.images = []
        path = data_dir
        settype = []
        settype.append(setname)
        self.classindex = {'amusement':0, 'anger':1, 'awe':2, 'contentment':3, 'disgust':4, 'excitement':5, 'fear':6, 'sadness':7}
        self.subclasssentence = ['ecstasy, joy, or serenity',
                                 'rage, anger, or annoyance',
                                 'amazement, surprise, or distraction',
                                 'admiration, trust, or acceptance',
                                 'loathing, disgust, or boredom',
                                 'vigilance, anticipation, or interest',
                                 'terror, fear, or apprehension',
                                 'grief, sadness, or pensiveness']

        for s in settype:
            for ct in classtype:
                filepath = path + s + '/' + ct + '/'
                file_list = os.listdir(filepath)
                for filename in file_list:
                    img_path = filepath + filename
                    img_set = s
                    img_label = ct
                    cate_all = []
                    # catepath = filename.strip(ct+'_').strip('jpg')+'txt'
                    # if catepath in catedir:
                    #     # img_cate = img_label + ' ' + catedir[catepath]
                    #     # img_cate = 'a photo contains ' + catedirnlp[catepath] + ', and it seems to express a feeling of ' + img_label + '.'
                    #     # img_cate = 'a photo contains ' + catedirnlp[catepath] + ', and it seems to express a feeling of ' + img_label
                    #     img_cate = 'a photo contains ' + catedirnlp[catepath] + ', and it seems to express some feelings like '
                    #     for sub in self.subclasssentence:
                    #         # subcate = 'a photo seems to express some feelings like ' + sub + '.'
                    #         # subcate = 'This is an affective image, which seems to express some feelings like ' + sub + '.'
                    #         # subcate = 'In the field of image emotion analysis, this is an affective image, which seems to express an emotion or a feeling of ' + sub + '.'
                    #         subcate = img_cate + sub
                    #         # subcate = img_cate + sub + '.'
                    #         # subcate = 'a photo seems to express a feeling of ' + sub + '.'
                    #         cate_all.append(subcate)
                    # else:
                    # img_cate = img_label
                    # img_cate = 'a photo seems to express a feeling of ' + img_label + '.'
                    # img_cate = 'a photo seems to express a feeling of ' + img_label
                    # img_cate = 'a photo seems to express some feelings like '
                    for sub in classtype:
                        # subcate = 'a photo seems to express some feelings like ' + sub + '.'
                        # subcate = 'This is an affective image, which seems to express some feelings like ' + sub + '.'
                        # subcate = 'In the field of image emotion analysis, this is an affective image, which seems to express an emotion or a feeling of ' + sub + '.'
                        subcate = initT + sub
                        # subcate = img_cate + sub
                        # subcate = img_cate + sub + '.'
                        # subcate = 'a photo seems to express a feeling of ' + sub + '.'
                        cate_all.append(subcate)
                    self.images.append((img_path, img_set, img_label, cate_all))

    def __getitem__(self, item):
        image, imgset, label, t_all = self.images[item]
        if template == 'label':
            # text = clip.tokenize(label)[0]
            texts = clip.tokenize(classtype)
        else:
            # text = clip.tokenize(cateinfo)[0]  # try different way
            texts = clip.tokenize(t_all)

        # img_i = pil_loader(image)
        # img = data_transforms[imgset](img_i)

        img = preprocess(Image.open(image))

        labelindex = self.classindex[label]

        return img, texts, labelindex

    def __len__(self):
        return len(self.images)


def get_features(traindataset, valdataset):
    topacc = 0.0
    bestepoch = 0
    bestval = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        warmcount = 0
        for images, texts, labels in tqdm(DataLoader(traindataset, batch_size=batch_size, shuffle=True)):
        # for images, text, texts, labels in tqdm(DataLoader(traindataset, batch_size=batch_size, shuffle=True)):
            sfpt.train()
            warmcount += 1
            images = images.to(device)
            # text = text.to(device)
            texts = texts[0].to(device)
            labels = labels.to(device)
            # if epoch >= warmupNum:
            #     optimizer.zero_grad()
            optimizer_fc.zero_grad()
            with torch.set_grad_enabled(True):
                logitsimg, logitstxt = sfpt(images, texts)
                # logitsimg2, logitstxt2 = sfpt(images, texts)
                # # bs_real = int(len(logitsimg)/2)*2
                # # outputs = F.softmax(logitsimg.float(), dim=-1)
                # # log_out = F.log_softmax(logitsimg.float(), dim=-1)
                # # outputsP = outputs[:bs_real:2]
                # # outputsQ = outputs[1::2]
                # # logoutP = log_out[:bs_real:2]
                # # logoutQ = log_out[1::2]
                # # loss1 = criterion(outputs, labels)
                # # loss2 = criterionKL(logoutP, outputsQ) + criterionKL(logoutQ, outputsP)
                # outputsP = F.softmax(logitsimg.float(), dim=-1)
                # outputsQ = F.softmax(logitsimg2.float(), dim=-1)
                # log_outP = F.log_softmax(logitsimg.float(), dim=-1)
                # log_outQ = F.log_softmax(logitsimg2.float(), dim=-1)
                # loss1 = 0.5 * (criterion(logitsimg.float(), labels) + criterion(logitsimg2.float(), labels))
                # loss2 = 0.5 * (criterionKL(log_outP, outputsQ) + criterionKL(log_outQ, outputsP))
                # loss = loss1 + loss2

                outputs = S(logitsimg.float())
                loss = criterion(outputs, labels)

                # confidencesP, predsP = torch.max(outputsP, 1)
                # confidencesQ, predsQ = torch.max(outputsQ, 1)
                confidences, preds = torch.max(outputs, 1)
                loss.backward()

                # if epoch >= warmupNum:
                #     optimizer.step()
                optimizer_fc.step()
            running_loss += loss.item() * logitsimg.size(0)
            # running_corrects += 0.5 * (torch.sum(predsP == labels.data) + torch.sum(predsQ == labels.data))
            running_corrects += torch.sum(preds == labels.data)
            # if warmcount % 50 == 0:
            #     get_features_test(testset)

        epoch_losstrain = running_loss / len(traindataset)
        epoch_acctrain = running_corrects.double() / len(traindataset)
        loginfo = 'Train Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_losstrain, epoch_acctrain)
        log.append(loginfo+'\n')
        print(loginfo)

        # if epoch >= warmupNum:
        #     scheduler.step()
        scheduler_fc.step()
        sfpt.eval()
        running_lossval = 0.0
        running_correctsval = 0
        # for images, text, texts, labels in tqdm(DataLoader(valdataset, batch_size=batch_size, shuffle=False)):
        for images, texts, labels in tqdm(DataLoader(valdataset, batch_size=batch_size, shuffle=False)):
            images = images.to(device)
            texts = texts[0].to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                logitsimg, logitstxt = sfpt(images, texts)
                outputsP = F.softmax(logitsimg.float(), dim=-1)
                loss = criterion(logitsimg.float(), labels)
                confidences, preds = torch.max(outputsP, 1)
            running_lossval += loss.item() * logitsimg.size(0)
            running_correctsval += torch.sum(preds == labels.data)
        epoch_lossval = running_lossval / len(valdataset)
        epoch_accval = running_correctsval.double() / len(valdataset)
        loginfo = 'Val Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_lossval, epoch_accval)
        if round(epoch_accval.item(), 6) > bestval:
            bestval = round(epoch_accval.item(), 6)
            bestepoch = epoch
            if epoch > stepsize_FC:
                # get_features_test(testset)
                if savemodel:
                # if epoch > 5 and epoch_accval > topacc:
                    topacc = epoch_accval
                    best_model = copy.deepcopy(sfpt.state_dict())
                    # best_fc = copy.deepcopy(fc.state_dict())
                    torch.save(best_model, 'BESTSOFTMODEL' + str(round(float(topacc.item()), 6)) + '.pt')
                    # torch.save(best_fc, 'BESTFC' + str(topacc.item()) + '.pt')
                    print('Model save!')
        log.append(loginfo+'\n')
        print(loginfo)
        # if epoch % 5 == 4:
        #     get_features_test(testset)
        get_features_test(testset)
    return bestepoch, round(bestval, 6)


def get_features_test(dataset):
    sfpt.eval()
    running_loss = 0.0
    running_corrects = 0
    testResult = [[0 for col in range(num_classes)] for row in range(num_classes)]
    with torch.no_grad():
        # for images, text, texts, labels in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
        for images, texts, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        # for images, texts, labels in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
            images = images.to(device)
            texts = texts[0].to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                logitsimg, logitstxt = sfpt(images, texts)
                outputsP = F.softmax(logitsimg.float(), dim=-1)
                loss = criterion(logitsimg.float(), labels)
                confidences, preds = torch.max(outputsP, 1)
            predTemp = preds.cpu().numpy().tolist()
            testTemp = labels.cpu().numpy().tolist()
            for index in range(len(testTemp)):
                testResult[testTemp[index]][predTemp[index]] += 1
            running_loss += loss.item() * logitsimg.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        print('Answers in testlist:', testResult)
        loginfo = 'Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc)
        log.append(loginfo+'\n')
        log.append(str(testResult))
        print(loginfo)

        return round(float(epoch_acc.cpu().detach().numpy()), 6)


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Linear(num_classes, num_classes)
        # self.classifier1 = nn.Linear(num_classes, 2*num_classes-1)
        # self.classifier2 = nn.Linear(2*num_classes-1, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = F.relu(self.classifier1(x))
        # x = self.classifier2(x)

        return x


class NEWALL(nn.Module):
    def __init__(self,
                 soft: bool
                 ):
        super().__init__()
        self.soft = soft
        self.context_length = model.context_length

        self.visual = model.visual

        self.transformer = model.transformer

        # self.vocab_size = vocab_size
        self.vocab_size = model.vocab_size

        # self.token_embedding = nn.Embedding(self.vocab_size, self.transformer.width)  # ori
        self.token_embedding = model.token_embedding
        if self.soft:
            # # # self.n_tokens = 40
            # # # self.n_tokens = 10
            # # # self.random_range = 0.5
            # # # self.initialize_from_vocab = False
            # # # self.initialize_from_vocab = True
            # # # tt = clip.tokenize('a photo seems to express a feeling of ')[0].to(device)
            # # tt = clip.tokenize('In the field of image emotion analysis, this is an affective image, which seems to express an emotion or a feeling of ')[0].to(device)
            # cate_all = []
            # # for sub in self.subclasssentence:
            # st = True
            # for sub in classtype:
            #     # subcate = 'a photo seems to express some feelings like ' + sub + '.'
            #     # subcate = 'This is an affective image, which seems to express some feelings like ' + sub + '.'
            #     # subcate = 'In the field of image emotion analysis, this is an affective image, which seems to express an emotion or a feeling of ' + sub + '.'
            #     # subcate = 'a photo seems to express some feelings like ' + sub + '.'
            #     subcate = 'a photo seems to express some feelings like '
            #
            #     cate_all.append(subcate)
            #     tt = clip.tokenize(subcate)[0].to(device)
            #
            #     if st:
            #         init_e = torch.index_select(self.token_embedding.weight, 0, tt).clone().detach()
            #         init_e = torch.unsqueeze(init_e, 0)
            #         st = False
            #     else:
            #         init_new = torch.index_select(self.token_embedding.weight, 0, tt).clone().detach()
            #         init_new = torch.unsqueeze(init_new, 0)
            #         init_e = torch.vstack((init_e, init_new))
            subcate = initT
            tt = clip.tokenize(subcate)[0].to(device)
            self.n_tokens = torch.max(tt, 0)[1].item()
            init_e = torch.index_select(self.token_embedding.weight, 0, tt).clone().detach()[:self.n_tokens]
            self.initialize_embedding = init_e.repeat(num_classes, 1, 1).clone().detach()

            # tt_all = clip.tokenize(cate_all).to(device)
            # self.initialize_embedding = init_e.clone().detach()

            # # tt = clip.tokenize('a photo seems to express some feelings like anything.')[0].to(device)
            # # tt = clip.tokenize('a photo seems to express some feelings like ')[0].to(device)
            # self.n_tokens = torch.max(tt, 0)[1].item()
            # tt[torch.max(tt, 0)[1].item()] = 0
            # # tt = clip.tokenize('a photo seems to express some feelings like ')[0][:self.n_tokens].to(device)
            # self.initialize_embedding = torch.index_select(self.token_embedding.weight, 0, tt).clone().detach()
            # self.initialize_embedding = torch.index_select(self.token_embedding.weight, 0, tt_all).clone().detach()
            # self.initialize_embedding = self.token_embedding.weight[:self.n_tokens].clone().detach()  # get top n_tokens' embedding
            # self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding[:self.n_tokens])
            self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding)

        # self.token_embedding = SoftEmbedding(self.token_embedding_ori, n_tokens=10,
        #                                      initialize_from_vocab=True)  # added by Snow
        self.positional_embedding = model.positional_embedding
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = model.ln_final
        # self.ln_final = clip.model.LayerNorm(transformer_width)

        self.text_projection = model.text_projection
        # self.text_projection = nn.Parameter(torch.empty(512, batch_size*num_classes))  ###
        self.logit_scale = model.logit_scale
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # nn.init.normal_(self.token_embedding.weight, std=0.02)  # ori
        # nn.init.normal_(self.positional_embedding, std=0.01)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, imgfeature):
        if not self.soft:
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        else:
            learned_embedding = self.learned_embedding.type(self.dtype)
            learned_norm = learned_embedding / learned_embedding.norm(dim=-1, keepdim=True)
            logits_per_prefix = learned_norm @ imgfeature.t()
            logits_inone = (logits_per_prefix.permute(2, 0, 1) + 1) / 2
            learntemp = learned_embedding.repeat(logits_per_prefix.shape[-1], 1, 1, 1).permute(3, 0, 1, 2)
            cal = torch.mul(learntemp, logits_inone)
            calend = torch.sum(cal, dim=2, keepdim=True)
            calend = calend.permute(2, 1, 3, 0).squeeze().repeat(num_classes, 1, 1, 1).permute(1, 0, 2, 3)
            input_embedding = self.token_embedding(text[:, self.n_tokens:]).type(self.dtype)
            x = torch.cat([calend, input_embedding.repeat(logits_per_prefix.shape[-1], 1, 1, 1)], 2)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.reshape(x.shape[0]*num_classes, 77, 512)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x.reshape(int(x.shape[0]/num_classes), num_classes, 77, 512)
        x = x.permute(1, 2, 0, 3)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        x = x.permute(1, 0, 2)
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.encode_text(text, image_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        image_features = torch.unsqueeze(image_features, -1)
        image_features = image_features / image_features.norm(dim=-2, keepdim=True)
        # logits_per_image = logit_scale * torch.bmm(image_features, text_features)
        logits_per_text = logit_scale * torch.bmm(text_features, image_features)
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logit_scale * text_features @ image_features.t()
        logits_per_text = logits_per_text.squeeze()

        return logits_per_text, logits_per_text
        # return logits_per_image, logits_per_text


if __name__ == '__main__':
    log = []
    t = time.strftime('%Y%m%d%H%M%S')
    print(t)
    dataset = data_dir.split('/')[-2]
    paraminfo = str(dataset)+' lr:'+str(lr_FC)+' stepsize:'+str(stepsize_FC)+' gamma'+str(gamma_FC)
    print(paraminfo)
    log.append(paraminfo + '\n')
    settinginfo = 'loadway: ' + str(loadbestmodel) + ' softchoice: ' + str(softchoice) + ' templateway: ' + str(template)
    print(settinginfo)
    log.append(settinginfo)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model, preprocess = clip.load('ViT-B/32', device)
    # print(model)
    starttime = time.strftime('%Y%m%d%H%M%S')
    print(starttime)

    # fc = FCN().to(device)
    if loadbestmodel == True:
        best_model = torch.load('BESTMODEL.pt')
        ## best_fc = torch.load('BESTFC.pt')
        ## fc.load_state_dict(best_fc)
        model = build_model(best_model).to(device)

    sfpt = NEWALL(soft=softchoice).to(device)

    # best_model = torch.load('BESTSOFTMODEL.pt')
    # for key in ["positional_embedding", "text_projection", "logit_scale"]:
    #     if key in best_model:
    #         del best_model[key]
    # sfpt.load_state_dict(best_model)

    params_to_update_fc = []

    for name, param in sfpt.named_parameters():
        if softchoice and 'learn' in name:
            params_to_update_fc.append(param)
            print("\t", name)

    print("params load finish")
    trainset = Loaddata('train')
    warmpos = 0.1 * len(trainset) / batch_size
    valset = Loaddata('val')
    testset = Loaddata('test')
    criterion = nn.CrossEntropyLoss()
    criterionKL = nn.KLDivLoss()
    S = nn.Softmax(dim=-1)
    if softchoice:
        optimizer_fc = optim.AdamW(params_to_update_fc, lr=lr_FC)
        scheduler_fc = lr_scheduler.StepLR(optimizer_fc, step_size=stepsize_FC, gamma=gamma_FC)

    if not onlytest:
        bestepoch, bestvalacc = get_features(trainset, valset)
        print('bestepoch: ' + str(bestepoch) + 'bestval: ' + str(bestvalacc))
    acc = 100 * get_features_test(testset)
    t = time.strftime('%Y%m%d%H%M%S')
    print(t)
    print(paraminfo)
    log.append(paraminfo+'\n')
    print(settinginfo)
    log.append(settinginfo)

    with open('./log/'+str(dataset)+str(acc)+'_'+str(t)+'.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')

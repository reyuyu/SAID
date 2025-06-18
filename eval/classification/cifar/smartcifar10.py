import torch
import sys
sys.path.append('../../..')
from model import longclip
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from eval.classification.cifar.templates import imagenet_templates



def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        unnorm_weights = []
        full_embeddings = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = longclip.tokenize(texts).cuda()  # tokenize
            class_embeddings, full_class_embeddings = model.encode_text(texts, return_full=True)  # embed with text encoder
            unnorm_weights.append(class_embeddings.mean(dim=0))
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            full_embeddings.append(full_class_embeddings.mean(dim=0))
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        unnorm_weights = torch.stack(unnorm_weights, dim=1).cuda()
        full_embeddings = torch.stack(full_embeddings, dim=0).cuda()
    return zeroshot_weights, unnorm_weights, full_embeddings

@torch.no_grad()
def eval_smartcifar10(model, preprocess, num_eval_samples=1000, soft_mask=False):
    try:
        model = model.module
        rank = torch.distributed.get_rank()
    except:
        rank = 0
        pass
    testset = torchvision.datasets.CIFAR10(root="/mnt/data/shaoanxi/clip_data/cifar10", train=False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)
    text_feature, unnorm_text_feature, full_embeddings = zeroshot_classifier(model, testset.classes, imagenet_templates)
    device = torch.device("cuda")
    correct = 0
    total = 0
    smart_correct = 0
    smart_mask_correct = 0
    smart_image_correct = 0
    with torch.no_grad():
        i = 0
        pbar = tqdm(testset, disable=(rank != 0))
        for data in pbar:
            images, labels = data
            images = preprocess(images).unsqueeze(0).to(device)

            unnorm_image_feature = model.encode_image(images)
            image_feature = unnorm_image_feature / unnorm_image_feature.norm(dim=-1, keepdim=True)

            smart_image_feature = image_feature.repeat(10, 1)
            with autocast():
                tmp_text_feature = text_feature.to(torch.float32)
                """
                smart_image_feature = smart_image_feature.to(torch.float32)
                z, _, log_det_xz = model.flow.flow_xz(smart_image_feature)
                mask_logit = model.mask_net((smart_image_feature * tmp_text_feature.T) * 100)
                soft_mask = (torch.tanh(mask_logit) + 1) / 2
                hard_mask = (soft_mask >= 0.5).float() - soft_mask.detach() + soft_mask
                masked_z = z * hard_mask + (1 - hard_mask) * model.flow_null_param
                smart_image_feature = model.flow.flow_xz.inverse(masked_z)[0]
                """
                smart_image_feature, hard_mask = model.compute_smart_image_feature(unnorm_image_feature, full_text_embedding=full_embeddings)[:2]

            smart_mask_sims = torch.sum(hard_mask, dim=1)
            smart_sims = torch.sum(smart_image_feature * text_feature.T, dim=-1).view(-1)
            smart_sims2 = torch.sum(smart_image_feature * image_feature, dim=-1).view(-1)
            smart_pred = torch.argmax(smart_sims, dim=0).item()
            smart_mask_pred = torch.argmax(smart_mask_sims, dim=0).item()
            vanilla_sim = image_feature @ text_feature

            # print(smart_mask_sims, smart_sims, smart_sims2, vanilla_sim, labels, smart_pred)
            # if total>10:
            #    raise Exception
            smart_image_pred = torch.argmax(smart_sims2, dim=0).item()
            if labels == smart_image_pred:
                smart_image_correct += 1

            if labels == smart_mask_pred:
                smart_mask_correct += 1

            if labels == smart_pred:
                smart_correct += 1

            sims = image_feature @ text_feature
            pred = torch.argmax(sims, dim=1).item()
            if labels == pred:
                correct += 1
            total += 1
            pbar.set_description("Mask %.2f Sim: %.2f Long: %.2f Image: %.2f NumAlive: %d" % (
                100 * smart_mask_correct / total, 100 * smart_correct / total, 100 * correct / total,
                100 * smart_image_correct / total, int(torch.sum(hard_mask, dim=1).mean().item())))
            if total >= num_eval_samples:
                break

        return {'smart_mask_correct': smart_mask_correct/total, 'smart_correct': smart_correct/total, 'correct': correct/total}


if __name__ == '__main__':
    from templates import imagenet_templates
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--nb_mix', type=int, default=1)
    parser.add_argument('--nb_latent', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(args.checkpoint, device=device, args=args)
    model.eval()
    eval_smartcifar10(model, preprocess)
    raise Exception


    testset = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

    text_feature = zeroshot_classifier(model, testset.classes, imagenet_templates)
    correct = 0
    total = 0
    smart_correct = 0
    smart_mask_correct = 0
    smart_image_correct = 0
    with torch.no_grad():
        i = 0
        pbar = tqdm(testset)
        for data in pbar:
            images, labels = data
            images = preprocess(images).unsqueeze(0).to(device)

            image_feature = model.encode_image(images)
            image_feature = image_feature/image_feature.norm(dim=-1, keepdim=True)

            smart_image_feature = image_feature.repeat(10, 1)
            with autocast():
                smart_image_feature = smart_image_feature.to(torch.float32)
                tmp_text_feature = text_feature.to(torch.float32)
                z, _, log_det_xz = model.flow.flow_xz(smart_image_feature)
                mask_logit = model.mask_net((smart_image_feature * tmp_text_feature.T)*100)
                soft_mask = (torch.tanh(mask_logit)+1)/2
                hard_mask = (soft_mask >= 0.5).float() - soft_mask.detach() + soft_mask
                masked_z = z * hard_mask + (1 - hard_mask) * model.flow_null_param
                smart_image_feature = model.flow.flow_xz.inverse(masked_z)[0]
                #smart_image_feature = model.compute_smart_image_feature(smart_image_feature, tmp_text_feature.T)

            smart_mask_sims = torch.sum(hard_mask, dim=1)
            smart_sims = torch.sum(smart_image_feature * text_feature.T, dim=-1).view(-1)
            smart_sims2 = torch.sum(smart_image_feature * image_feature, dim=-1).view(-1)
            smart_pred = torch.argmax(smart_sims, dim=0).item()
            smart_mask_pred = torch.argmax(smart_mask_sims, dim=0).item()
            vanilla_sim = image_feature @ text_feature
            #print(smart_mask_sims, smart_sims, smart_sims2, vanilla_sim, labels, smart_pred)
            #if total>10:
            #    raise Exception
            smart_image_pred = torch.argmax(smart_sims2, dim=0).item()
            if labels == smart_image_pred:
                smart_image_correct += 1

            if labels == smart_mask_pred:
                smart_mask_correct += 1

            if labels == smart_pred:
                smart_correct += 1

            sims = image_feature @ text_feature
            pred = torch.argmax(sims, dim=1).item()
            if labels == pred:
                correct += 1
            total += 1
            pbar.set_description("Mask %.2f Sim: %.2f Long: %.2f Image: %.2f" % (100*smart_mask_correct/total, 100*smart_correct/total, 100*correct/total, 100*smart_image_correct/total))

        print(smart_mask_correct)
        print(smart_mask_correct/total)
        print(smart_correct)
        print(smart_correct/total)
        print(correct)
        print(total)
        print(correct/total)

    print("Accuracy of the LongCLIP model on the CIFAR-10 test images: %d %%" % (100 * correct / total))
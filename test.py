from model.samclip_predictor import SAMCLIP
import torch

def test():
    model_2d = SAMCLIP("./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px")
    pixel_feature = model_2d.extract_image_feature('../project/data/lerf_ovs/waldo_kitchen/images/00066.jpg')
    text_feature=model_2d.extract_text_feature(['plastic ladle', 'pot', 'refrigerator', 'spatula'])
    torch.save(pixel_feature, 'pixel_feature.pt')
    torch.save(text_feature, 'text_feature.pt')

def main():
    test()

if __name__ == '__main__':
    main()
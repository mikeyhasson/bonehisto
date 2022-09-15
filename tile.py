from sahi.slicing import slice_coco
from sahi.utils.file import load_json

for phase in ["train", "valid", "test"]:
    src = 'resnet18fpn/rfimg/' + phase
    coco_dict = load_json(src + "/_annotations.coco.json")

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=src + "/_annotations.coco.json",
        image_dir=src,
        output_coco_annotation_file_name="_annotations",
        ignore_negative_samples=False,
        output_dir="resnet18fpn/bonecellnew/" + phase,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.15,
        overlap_width_ratio=0.15,
        min_area_ratio=0.2,
        verbose=True
    )

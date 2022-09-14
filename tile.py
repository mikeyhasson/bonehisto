from sahi.slicing import slice_coco
from sahi.utils.file import load_json
for phase in ["train", "valid", "test"]:

    src = 'rfimg/' +phase
    coco_dict = load_json(src+"/_annotations.coco.json")

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=src+"/_annotations.coco.json",
        image_dir=src,
        output_coco_annotation_file_name="_annotations",
        ignore_negative_samples=False,
        output_dir="bonecell/"+phase,
        slice_height=224,
        slice_width=224,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        min_area_ratio=0.2,
        verbose=True
    )
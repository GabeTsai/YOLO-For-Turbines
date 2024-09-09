loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = False)
    
    x, y = next(iter(loader))
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_boxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=0.5, obj_threshold= 0.7, box_format="midpoint",
        )
        plot_image_with_boxes(x[i].permute(1,2,0).detach().cpu(), nms_boxes)

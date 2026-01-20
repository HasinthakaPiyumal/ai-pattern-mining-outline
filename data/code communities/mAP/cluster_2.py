# Cluster 2

def error(msg):
    print(msg)
    sys.exit(0)

def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return (img, line_width + text_width)

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    '\n     This part makes the precision monotonically decreasing\n        (goes from the end to the beginning)\n        matlab: for i=numel(mpre)-1:-1:1\n                    mpre(i)=max(mpre(i),mpre(i+1));\n    '
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    '\n     This part creates a list of indexes where the recall changes\n        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;\n    '
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    '\n     The Average Precision (AP) is the area under the curve\n        (numerical integration)\n        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));\n    '
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return (ap, mrec, mpre)


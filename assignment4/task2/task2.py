import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes
import operator
import pdb # for debuggin purposes

def calculate_iou(prediction_box, gt_box):

    if prediction_box[0]>gt_box[2] or prediction_box[2]< gt_box[0] or prediction_box[1]>gt_box[3] or prediction_box[3]< gt_box[1] :
        intersection = 0
    else :
        box_inter =[]
        for k in range (len(gt_box)):
            if k<2:
                box_inter.append(max(prediction_box[k],gt_box[k]))
            else :
                box_inter.append(min(prediction_box[k],gt_box[k]))



        intersection = (box_inter[2]-box_inter[0])*(box_inter[3]-box_inter[1])
    """
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]


    """

    
    sum_box = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) + (prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    union=sum_box - intersection
    
    iou = intersection/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):

    total_d = (num_tp+num_fp)
    if total_d == 0:
        return 1
    else :
        prec = num_tp / total_d
        return prec



def calculate_recall(num_tp, num_fp, num_fn):
    total_case = num_tp+num_fn
    if total_case ==0:
        return 0
    else:
        recall = num_tp/ total_case
        return recall



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):

        
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    

    if  len(prediction_boxes)==0 or len(gt_boxes)==0 :
        return np.array([]), np.array([])

    m=gt_boxes.shape[0]
    n=gt_boxes.shape[1]
    count=0
    dict_IoU ={}
    matched_box =np.zeros((m,n))
    predicted_box=np.zeros((m,n))
    for i in range(gt_boxes.shape[0]):
        for k in range(prediction_boxes.shape[0]) :
            iou_real = calculate_iou(prediction_boxes[k,:],gt_boxes[i,:])
            if iou_real>= iou_threshold :
                dict_IoU[(i,k)]= iou_real
    while dict_IoU:
        best_match = max(dict_IoU.keys(), key=lambda key: dict_IoU[key])
        temp = dict_IoU.copy()
        for key in temp.keys():
            if key[0]==best_match[0] or key[1]==best_match[1]:
                dict_IoU.pop(key)
        matched_box[count,:] = gt_boxes[best_match[0],:]
        predicted_box[count,:] = prediction_boxes[best_match[1],:]
        count+=1
    if m!=count :
        index = m-count
        return predicted_box[:-index, :],matched_box[:-index, :]
    else:
        return predicted_box,matched_box



def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    _predicted, _matched = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    values = {
        "true_pos": len(_predicted), 
        "false_pos": len(prediction_boxes) - len(_predicted),
        "false_neg": len(gt_boxes) - len(_predicted)
    }
    return values

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    num_tp = 0
    num_fp = 0
    num_fn = 0
    for p, gt in zip(all_prediction_boxes, all_gt_boxes):
        values = calculate_individual_image_result(p, gt, iou_threshold)
        num_tp += values["true_pos"]
        num_fp += values["false_pos"]
        num_fn += values["false_neg"]
    precision = calculate_precision(num_tp, num_fp, num_fn)
    recall = calculate_recall(num_tp, num_fp, num_fn)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.
    Args:
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    precisions = []
    recalls = []
    confidence_thresholds = np.linspace(0, 1, 500)
    for thresh in np.nditer(confidence_thresholds):
        _prediction_boxes = []
        for i in range(len(confidence_scores)):
            index=[]
            for j in range(len(confidence_scores[i])):
                if confidence_scores[i][j] > thresh:
                    # delete from predictions
                    index.append(j)
            p_image = np.zeros((len(index),4))
            for row in range(len(index)):
                p_image[row,:] = all_prediction_boxes[i][index[row]]
            _prediction_boxes.append(p_image)

        p, r = calculate_precision_recall_all_images(_prediction_boxes, all_gt_boxes, iou_threshold)
        precisions.append(p)
        recalls.append(r)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
   
    L=[]
    recall_levels = np.linspace(0, 1.0, 11)
    for k in range(len(recall_levels)) :
        maxi=0
        for i in range(len(recalls)):
            if recall_levels[k]<= recalls[i] and maxi < precisions[i]:
                maxi = precisions[i]
        L.append(maxi)
    average_precision=sum(L)/len(L)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

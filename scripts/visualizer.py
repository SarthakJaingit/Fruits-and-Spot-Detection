import matplotlib.pyplot as plt
import numpy as np
import cv2
import train


def visualize_images(loader):
    print("Showing Two images")
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(2):
        ax = fig.add_subplot(2, 2/2, idx+1, xticks=[], yticks=[])
        image = draw_boxes(labels[idx]["boxes"], labels[idx]["labels"], images[idx])
        plt.imshow(image)
        plt.show(block = True)

def draw_boxes(boxes, labels, image, infer = False, put_text = True):
    COLORS = [(0, 0, 0), (0, 255, 0), (0, 0 , 255), (255, 255, 0), (255, 0, 0)]
    classes = ["Placeholder", "Apples", "Strawberry", "Apple_Bad_Spot", "Strawberry_Bad_Spot"]
    # read the image with OpenCV
    image = image.permute(1, 2, 0).numpy()
    if infer:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i] % len(COLORS)]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        if put_text:
          cv2.putText(image, classes[labels[i]], (int(box[0]), int(box[1]-5)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                      lineType=cv2.LINE_AA)
    return image


if __name__ == "__main__":

    train_loader, test_loader = train.create_dataloaders(prep_noiseloader = False)
    visualize_images(train_loader)

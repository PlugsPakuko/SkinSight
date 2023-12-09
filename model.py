from ultralytics import YOLO
# Run inference

# load model
burn_model = YOLO('./weights/SkinBurn.pt')
abrasion_model = YOLO('./weights/abrasion.pt')
cut_model = YOLO('./weights/Cut.pt')

test = 'https://www.collinsdictionary.com/images/full/hand_large_702121612_1000.jpg'

#test
def predict(INPUT, MODEL):
    if(MODEL == 'abrasion'):
        results = abrasion_model(INPUT)
    elif(MODEL == 'burn'):
        results = burn_model(INPUT)
    elif(MODEL == 'cut'):
        results = cut_model(INPUT)
    return [float(results[0].probs.top1conf * 100), results[0].probs.top1 + 1]

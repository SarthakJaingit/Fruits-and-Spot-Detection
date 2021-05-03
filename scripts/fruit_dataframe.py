import pandas as pd
import re

def run_dataframe_check(fruit_df, list_of_all_dataframes):
    total_row_sum_check = 0
    for dataframe in list_of_all_dataframes:
        total_row_sum_check += dataframe.shape[0]

    assert total_row_sum_check == fruit_df.shape[0]
    print("Checked Loaded Dataset -> Passed Assertion")
    print("DataFrame shape: {}".format(fruit_df.shape))
    print("Unique Fruit Labels {}".format(fruit_df["Fruit"].unique()))
    print("Number of Unique Images {}".format(len(fruit_df["Image_id"].unique())))

def more_specific_Image_id(image_id, fruit):
  if fruit == "Bad_Spots":
    if re.search("RottenStrawberries", image_id):
      return "Strawberry_Bad_Spot"
    elif re.search("RottenApples", image_id):
      return "Apple_Bad_Spot"
    else:
      raise ValueError("Could not find a match for some of the Image_ids")

  else:
    return fruit

def get_dataset(more_specific_spots = True):

    strawberry_csv_batch_3 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/FreshStrawberries/Fresh StrawBerry Batch 3 Labeled/FreshStrawberryBatch3Labels.csv", header = None)
    strawberry_csv_batch_2 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/FreshStrawberries/Fresh StrawBerry Batch 2 Labeled/FreshStrawberriesBatch2Labels.csv", header = None)
    strawberry_csv_batch_1 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/FreshStrawberries/Fresh StrawBerry Batch 1 Labeled/Strawberrybatch1.csv", header = None)
    rottenApple_csv_batch_1 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenApples/RottenAppleBatch1Labeled/RottenAppleBatch1Labels.csv", header = None)
    rottenApple_csv_batch_2 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenApples/RottenAppleBatch2Labeled/RottenApplesBatch2Labels.csv", header = None)
    rottenApple_csv_batch_3 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenApples/RottenAppleBatch3Labaled/RottenApplesBatch3Labels.csv", header = None)
    rottenStrawberry_csv_batch_1 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenStrawberries/Batch1RottenStrawBerryLabels/RottenStrawberriesBatch1Labels.csv", header = None)
    rottenStrawberry_csv_batch_2 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenStrawberries/Batch2RottenStrawBerryLabels/RottenStrawBerryBatch2.csv", header = None)
    rottenStrawberry_csv_batch_3 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/RottenStrawberries/Batch3RottenStrawberrylabel/rottenStrawberryBtch3labels.csv", header = None)
    freshApples_csv_batch_2 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/FreshApples/FreshApplebtch2label/FreshApplesBatch2LabelsFresh.csv", header = None)
    freshApples_csv_batch_1 = pd.read_csv("../ScriptFruitDet/ScriptDataset/Train/FreshApples/FreshApplesBatch1Labels/FreshAppleBatch1Labels.csv", header = None)

    strawberry_csv_batch_3.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    strawberry_csv_batch_2.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    strawberry_csv_batch_1.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenApple_csv_batch_1.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenApple_csv_batch_2.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenApple_csv_batch_3.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenStrawberry_csv_batch_1.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenStrawberry_csv_batch_2.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    rottenStrawberry_csv_batch_3.columns = ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    freshApples_csv_batch_2.columns =  ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]
    freshApples_csv_batch_1.columns =  ["Fruit", "Coord1", "Coord2", "Coord3", "Coord4", "Image_id", "OneSize", "TwoSize"]

#Drop some watermark data for Fresh StrawBerry Batch 1 Labeled images [59, 9, 93]

    strawberry_csv_batch_1.drop(strawberry_csv_batch_1[strawberry_csv_batch_1["Image_id"] == "FreshStrawberries59.jpeg"].index, inplace = True)
    strawberry_csv_batch_1.drop(strawberry_csv_batch_1[strawberry_csv_batch_1["Image_id"] == "FreshStrawberries9.jpeg"].index, inplace = True)
    strawberry_csv_batch_1.drop(strawberry_csv_batch_1[strawberry_csv_batch_1["Image_id"] == "FreshStrawberries93.jpeg"].index, inplace = True)
    strawberry_csv_batch_1 = strawberry_csv_batch_1.reset_index(drop=True)

#Stack all the csv files together.
    list_of_all_dataframes = [strawberry_csv_batch_1, strawberry_csv_batch_2, strawberry_csv_batch_3, rottenApple_csv_batch_1,
    rottenApple_csv_batch_2, rottenApple_csv_batch_3, rottenStrawberry_csv_batch_1, rottenStrawberry_csv_batch_2,
    rottenStrawberry_csv_batch_3, freshApples_csv_batch_2, freshApples_csv_batch_1]

    fruit_df = pd.concat(list_of_all_dataframes, ignore_index = True)

    if more_specific_spots:
        fruit_df["Fruit"] = fruit_df.apply(lambda row: more_specific_Image_id(row.Image_id, row.Fruit), axis = 1)

    run_dataframe_check(fruit_df, list_of_all_dataframes)

    #Extra Post Processing
    fruit_df = fruit_df[fruit_df["Image_id"] != "FreshStrawberries15.jpeg"]

    return fruit_df

def get_dict(classes):

    fruit_df = get_dataset()
    bounding_box_dict, labels_dict = dict(), dict()

    for row_index in range(len(fruit_df)):
        current_image_file = fruit_df.iloc[row_index]["Image_id"]
        if current_image_file not in bounding_box_dict:
            bounding_box_dict[current_image_file] = list()
            labels_dict[current_image_file] = list()
        bounding_box_dict[current_image_file].append(fruit_df.iloc[row_index, 1:5].to_list())
        labels_dict[current_image_file].append(classes.index(fruit_df.iloc[row_index, 0]))

    return bounding_box_dict, labels_dict

if __name__ == "__main__":

    fruit_df = get_dataset()
    bounding_box_dict, labels_dict = get_dict(["Placeholder", "Apples", "Strawberry", "Apple_Bad_Spot", "Strawberry_Bad_Spot"])

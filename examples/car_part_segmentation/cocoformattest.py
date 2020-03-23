from pathlib import Path
import argparse
import json
from datetime import datetime
from enum import Enum


#base_path = Path(__file__).parent.parent
#path_cogito_json = base_path / Path('tests/cogito_output_merged.json')


class TipoAnnotazione(Enum):
    object_detection = 1
    keypoint_detection = 2
    panoptic_segmentation = 3

class AllowedCocoData():
    

    coco_dict = {
        "info": {
            "year": int, 
            "version": str, 
            "description": str, 
            "contributor": str, 
            "url": str, 
            "date_created": type(datetime.strptime('2000/01/01', '%Y/%m/%d'))
        },
        "licenses": {
            "id": int, 
            "name": str, 
            "url": str
        },
        "images": {
            "id": int, 
            "width": int, 
            "height": int, 
            "file_name": str, 
            "license": int, 
            "flickr_url": str, 
            "coco_url": str, 
            "date_captured": type(datetime.strptime("2013/11/14 17:02:52", '%Y/%m/%d %H:%M:%S'))
        }
    }

    obj_det_annotation = {
        "annotations":{
            "id": int, 
            "image_id": int, 
            "category_id": int, 
            "segmentation": list, #RLE or [polygon], 
            "area": float, 
            "bbox": list, #[x,y,width,height], 
            "iscrowd": [0, 1]
            },
        "categories":{
            "id": int, 
            "name": str, 
            "supercategory": str
            }
    }
    keypoint_annotation = {
        "annotations" : {
            "keypoints": list, 
            "num_keypoints": int
        },
        "categories" : {
            "keypoints": [str], 
            "skeleton": list, #[edge]
        }
    }
    panoptic_annotation = {
        "annotations" : {
            "image_id": int, 
            "file_name": str, 
            "segments_info": list, #[segment_info]
        },
        "segment_info" : {
            "id": int,
            "category_id": int, 
            "area": int, 
            "bbox": list, #[x,y,width,height], 
            "iscrowd": [0, 1]
        },
        "categories": {
            "id": int, 
            "name": str, 
            "supercategory": str, 
            "isthing": 0 or 1, 
            "color": list, #[R,G,B]
        }
    }


    def __init__(self, jsonfile, tipo_annotazione = 1):
        with open(jsonfile) as json_file:
            self.data = json.load(json_file)
        if tipo_annotazione == TipoAnnotazione.object_detection.value:
            print("\n Verifica con annotazione 'Object detection'...\n")
            self.coco_dict["annotations"] = self.obj_det_annotation["annotations"] 
            self.coco_dict["categories"] = self.obj_det_annotation["categories"]
        elif tipo_annotazione == TipoAnnotazione.keypoint_detection.value:
            print("\nverifica con annotazione 'Keypoint detection'...\n")
            self.coco_dict["annotations"] = self.keypoint_annotation["annotations"] 
            self.coco_dict["categories"] = self.keypoint_annotation["categories"]
        elif tipo_annotazione == TipoAnnotazione.panoptic_segmentation.value:
            print("\nverifica con annotazione ' Panoptic segmentation'...\n")
            self.coco_dict["annotations"] = self.panoptic_annotation["annotations"] 
            self.coco_dict["categories"] = self.panoptic_annotation["categories"]
            self.coco_dict["segment_info"] = self.panoptic_annotation["segment_info"]
        else:
            print("\nNessuna verifica\n")

        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']        
        return


    def check_coco_format(self):
        #print(self.coco_dict.keys())
        if (set(self.data.keys()) != set(self.coco_dict.keys()) ):
            print('Chiavi al primo livello mancanti')
        
        for ch in self.coco_dict.keys():
            print(f'Check chiave {ch}')
            valore = self.data[ch]
            if(type(valore) != list):
                valore = [valore]
            i=0
            for v in valore:
                if (set(v.keys()) != set(self.coco_dict[ch].keys())):
                    print(f'Chiavi interne al livello {ch} mancanti')
            

                for k in v.keys():
                    if k not in self.coco_dict[ch].keys(): 
                        print(f'{k} è una chiave non Coco')
                    else:  
                        if (k == 'date_created' or k == 'date_captured'): #check sulla data
                            if (len(str(v[k])) == 0): 
                                i+=1
                                #print(f'Data in {ch}, posizione {i}, sottochiave {k} mamncante')
                            else:
                                try:
                                    tmpdate = datetime.strptime(v[k], '%Y/%m/%d %H:%M:%S')
                                except:
                                    tmpdate = datetime.strptime(v[k], '%Y/%m/%d')
                                    
                                if(tmpdate):
                                    if(type(tmpdate) != self.coco_dict[ch][k]):
                                        print('la data non è presente o non è in formato corretto')
                        
                        elif (k == 'iscrowd'):
                            if (v[k] not in self.coco_dict[ch][k]):
                                print('Valore iscrowd diverso da 0 o da 1')

                        elif (k == 'area'):
                            v[k] = float(v[k])

                        elif type(v[k]) != self.coco_dict[ch][k]: 
                            print(f'Il tipo di {k} non è quello giusto di Coco')
                            print(v[k])        

                        elif (k == 'image_id'):
                            if (len(str(v[k]).strip()) < 1 ):
                                print('!!! Image_id vuoto!')
            
            if ( i > 0): print(f'{i} date mancanti in {ch}')

        
        return

    def check_id_images(self):
        '''
        Controlla che per ogni Id contenuto in images esista almeno una 
        annotation e che ogni annotation abbia una imageId tra gli Id di images:
        -> Verifica che le imageIds e le id img prese da annotation siano uguale
        '''
        images_ids = [img['id'] for img in self.images]
        imgids_from_annotations = [ann['image_id'] for ann in self.annotations]

        print(f'Ci sono {len(images_ids)} Id immagini')
        #controlla chiavi ripetute
        if(len(set(images_ids)) != len(images_ids)):
            print(f'Ci sono {len(set(images_ids))} Id unici' )
        
        if (set(images_ids) != set(imgids_from_annotations)):
            not_common = set(images_ids) ^ set(imgids_from_annotations)
            if(len(set(images_ids)) > len(set(imgids_from_annotations))):
                print(f'Ci sono immagini senza annotazione: {not_common} ')
            else:
                print(f'Ci sono annotazioni con Id immagine non contenuti in \'images\' : {not_common} ')
    
    def check_annotations(self):
        '''
            Verifica che gli id categoria presi da categories e gli id category presi da annotation siano uguale
        '''
        id_cat_name = {cat['id']:cat['name'] for cat in self.categories}
        catids_from_annotations = [ann['category_id'] for ann in self.annotations]

        print(f'Ci sono {len(id_cat_name.items())} Id categoria')
        if(len(set(id_cat_name.keys())) != len(id_cat_name.keys())):
            print(f'Ci sono {len(set(id_cat_name.keys()))} Id unici' )

        for k,v in id_cat_name.items():
            print(f'Id {k}\t{v}')
        
        if (set(id_cat_name.keys()) != set(catids_from_annotations)):
            not_common = set(id_cat_name.keys()) ^ set(catids_from_annotations)
            if(len(set(id_cat_name.keys())) > len(set(catids_from_annotations))):
                print(f'Ci sono categorie mai annotate: {not_common} ')
            else:
                print(f'Ci sono annotazioni con Id categoria non contenuti in \'categories\' : {not_common} ')
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help="the filename to check",
                        default=None,
                        required=True)

    args = parser.parse_args()
    checker = AllowedCocoData(args.input)
    checker.check_coco_format()
    checker.check_id_images()
    checker.check_annotations()



import random
import collections
import math



class BalanceDataset():
    '''
        Class for balancing every category among train, validation, test sets
    Args:
        from_images_to_annotations: dictionary to obtain annotations contained in every image
        all_images_ids: all images Id
        from_categories_to_images: dictionary to obtain images containing such segment category

    '''

    def __init__(self, from_images_to_annotations, all_images_ids, from_categories_to_images):
        self.imgToAnns = from_images_to_annotations
        self.all_img_ids = all_images_ids
        self.catToImgs = from_categories_to_images

        self.train_p = None
        self.val_p = None
        self.test_p = None
        

    def getCatId_fromImgId(self, single_image_id: int) -> list:
        '''
            Get all segments categories contained in image with such Id
        '''
        return [d['category_id'] for d in self.imgToAnns[single_image_id]]

    def getCatId_from_many_imgsId(self, img_ids):
        '''
            Restituisce un dictionary che ha come chiavi l'Id categoria e come valori il numero di segmenti 
            di ciascuna categoria presenti nelle Id immagini input
        '''
        num_cat = collections.defaultdict(int) #zero default value
        for id in img_ids:
            cats_in_img = self.getCatId_fromImgId(id)
            for cat in cats_in_img:
                num_cat[cat] += 1
        return num_cat

    def dist_tot(self, img_ids, num_categories_target):
        num_cat = self.getCatId_from_many_imgsId(img_ids)
        sq_dist = [(num_cat[k] - v)*(num_cat[k] - v) for k,v in num_categories_target.items()]
        return math.sqrt(sum(sq_dist))

    def add_img(self, trial_set, remaining_set, to_add):
        trial_set.append(to_add)
        remaining_set.remove(to_add)
    def rem_img(self,trial_set, remaining_set, to_remove):
        trial_set.remove(to_remove)
        remaining_set.append(to_remove)
    def add_many_imgs(self, trial_set, remaining_set, lista):
        for l in lista:
            self.add_img(trial_set, remaining_set, l)
    def rem_many_imgs(self, trial_set, remaining_set, lista):
        for l in lista:
            self.rem_img(trial_set, remaining_set, l)


    def split_dataset_balanced_once(self, all_img_ids = None, set_perc=0.7):

        if all_img_ids is None:
            all_img_ids = self.all_img_ids

        num_categories_target = {cat_id:round(len(images)*set_perc) for cat_id, images in self.catToImgs.items()}

        #parto con un insieme iniziale
         
        total_imgs = len(all_img_ids)
        num_start_set = round(set_perc * total_imgs)
        start_imgs = random.sample(all_img_ids, num_start_set)

        add_trials = 4
        remove_trials = 4
        tot_trials = 200
        best_change = {}
        #epsilon = 0.01
        distances = {}
        min_over_dist = 0
        new_min = 0
        trial_set = start_imgs.copy()
        #remaining_set = list(filter(lambda x: x in start_imgs, all_img_ids))
        remaining_set = [x for x in all_img_ids if x not in start_imgs]

        for t in range(tot_trials):
    
            # Select the changes of previous iteration
            if (new_min != min_over_dist):
                min_over_dist = new_min
                #print(f'Il minimo è {new_min:.2f}')
                best_change[min_over_dist] = distances[min_over_dist]
                add_rem = best_change.get(min_over_dist)        
                self.add_img(trial_set, remaining_set, add_rem[0])
                self.rem_img(trial_set, remaining_set, add_rem[1])

            start_imgs = trial_set.copy()
            remaining_copy = remaining_set.copy()

            added = []
            for _ in range(add_trials):
                to_add = random.choice(remaining_set)
                self.add_img(trial_set, remaining_set, to_add)
                added.append(to_add)
                #dist = dist_tot(trial_set, imgToAnns, num_categories_target)
                #distances[dist] = (ai, None)

                removed = []
                for _ in range(remove_trials):
                    to_remove = random.choice(trial_set)
                    self.rem_img(trial_set, remaining_set, to_remove)
                    removed.append(to_remove)

                    dist = self.dist_tot(trial_set, num_categories_target)
                    distances[dist] = (to_add, to_remove)
                    #print(f'Distanza: {dist:.2f}')

                self.add_many_imgs(trial_set, remaining_set, removed)            

            self.rem_many_imgs(trial_set, remaining_set, added) 

            assert collections.Counter(trial_set) == collections.Counter(start_imgs)
            assert collections.Counter(remaining_copy) == collections.Counter(remaining_set)
                
            new_min = min(distances.keys())

        return trial_set, remaining_set
    
    def split_balanced(self):
        
        self.img_id_train, remaining_set = self.split_dataset_balanced_once(set_perc=self.train_p)
        remaining_frac = self.val_p / (1 - self.train_p)
        self.img_id_val, self.img_id_test = self.split_dataset_balanced_once(all_img_ids=remaining_set, set_perc=remaining_frac)

        print(f'Immagini di training: {len(self.img_id_train)}')
        print(f'Immagini di validation: {len(self.img_id_val)}')
        print(f'Immagini di test: {len(self.img_id_test)}')
        return self.img_id_train, self.img_id_val, self.img_id_test

    def set_percentages(self, train_percentage, val_percentage, test_percentage):
        self.train_p = train_percentage
        self.val_p = val_percentage
        self.test_p = test_percentage

        
    def verify_split(self):
        num_categories_tot = {cat_id:len(images) for cat_id, images in self.catToImgs.items()}
        num_categories_tot = {k: v for k, v in sorted(num_categories_tot.items(), key=lambda x: x[1], reverse=True)}
        #validation_categories_target = {cat_id:round(len(images)*self.val_p) for cat_id, images in catToImgs.items()}
        #test__categories_target = {cat_id:round(len(images)*self.test_p) for cat_id, images in catToImgs.items()}

        train_categories = self.getCatId_from_many_imgsId(self.img_id_train)
        val_categories = self.getCatId_from_many_imgsId(self.img_id_val)
        test_categories = self.getCatId_from_many_imgsId(self.img_id_test)

        print('{:<6s}{:>5s}\t{:<6s} {:<6s} {:<6s} {:<6s} {:<6s} {:<6s}'
        .format("CatId", "Tot", "train", "%", "valid","%", "test","%"))
        for c in num_categories_tot.items():
            cat, num = c
            #print(f'CatId {cat}, Tot:{num}, \
            #    train_p: {100 * train_categories[cat] / num :.1f} \
            #    val_p: {100 * val_categories[cat] / num :.1f} \
            #    test_p: {100 * test_categories[cat] / num :.1f} \
            #    ')
            print('{:<6}{:>5}\t{:<6}{:<6.1f} {:<6} {:<6.1f} {:<6} {:<6.1f}'
            .format(cat, num, 
            train_categories[cat],
            100 * train_categories[cat] / num, 
            val_categories[cat],
            100 * val_categories[cat] / num, 
            test_categories[cat],
            100 * test_categories[cat] / num))
        


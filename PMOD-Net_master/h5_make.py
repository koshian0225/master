import argparse
import h5py
import os
from h5datacreator import *
from rich.progress import track

def create_list_with_zeroes(length):
    values_list = [i for i in range(length)]
    return values_list

def extract_elements(values, start_index, interval, num_elements):
    extracted_elements = []
    for _ in range(num_elements):
        extracted_elements.append(values[start_index])
        del values[start_index]
        start_index += interval - 1
    return extracted_elements

def hdf5_input(file, train_file, val_file, test_file, train_frame, val_frame):
    with h5py.File(file,'r') as h5_data: 
        length = h5_data['header']['length'][()]
        values = create_list_with_zeroes(length)
        for i in range(2):
            if i == 0:num_elements = train_frame
            if i == 1:num_elements = val_frame
            interval = int((len(values) ) / num_elements)

            extracted_elements = extract_elements(values, 0, interval, num_elements)

            if i == 0:train_list = extracted_elements
            else :val_list = extracted_elements

        h5_main = h5_data.keys()

        image_name = h5_data['intrinsic'].keys()
        intrinsic_key = h5_data['intrinsic']['image_00'].keys()

        label = h5_data['label']['kitti360'].keys()

        tf = h5_data['tf_static'].keys()

        map = h5_data['map']['map'][()]

        map_attrs = h5_data['map/map'].attrs.keys()
        with h5py.File(train_file,'w') as h5_create:  
            for key in h5_main:
                h5_create.create_group(key)
                if key == 'data':
                    num=0
                    for i in track(train_list):
                        image_tag = h5_data[key][str(i)].keys()
                        h5_create.create_group(f'{key}/{num}')
                        data_path = h5_create[key][str(num)]
                        num+=1
                        for tag in image_tag:
                            if tag == 'oxts' or tag == 'world_to_pose':
                                img_group = data_path.create_group(tag)
                                img_keys = h5_data[key][str(i)][tag].keys()
                                img_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in img_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_group.attrs[attribute] = data
                                for img_name in img_keys:
                                    image = h5_data[key][str(i)][tag][img_name][()]
                                    img_dataset = data_path[tag].create_dataset(img_name,data=image)
                                    
                                    img_attrs = h5_data[key][str(i)][tag][img_name].attrs.keys()
                                    for attribute in img_attrs:
                                        data = h5_data[key][str(i)][tag][img_name].attrs[attribute]
                                        img_dataset.attrs[attribute] = data
                            elif tag == 'odometry':
                                continue
                            else:
                                image = h5_data[key][str(i)][tag][()]
                                img_dataset = data_path.create_dataset(tag,data=image)
                                image_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in image_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_dataset.attrs[attribute] = data
                if key == 'header':
                    h5_header = h5_create[key]
                    h5_header.create_dataset('length',data = length)
                elif key == 'intrinsic':            
                    for name in image_name:
                        intrinsic_name = os.path.join('intrinsic',name)
                        intrinsic_group = h5_create.create_group(intrinsic_name)
                        intrinsic_attrs = h5_data[intrinsic_name].attrs.keys()
                        for attribute in intrinsic_attrs:
                            data = h5_data[key][name].attrs[attribute]
                            intrinsic_group.attrs[attribute] = data
                        for intrinsic in intrinsic_key:
                            intrinsic_data = h5_data[key][name][intrinsic][()]
                            h5_intrinsic = h5_create[key][name]
                            h5_intrinsic.create_dataset(intrinsic,data=intrinsic_data)
                elif key == 'label':
                    h5_create.create_group('label/kitti360')
                    for name in label:
                        label_number = os.path.join(f'{key}/kitti360',name)
                        h5_create.create_group(label_number)
                        label_key = h5_data[label_number].keys()
                        label_path = h5_create[label_number]
                        for seg in label_key:
                            label_data = h5_data[label_number][seg][()]
                            label_path.create_dataset(seg,data=label_data)
                            if seg == 'color':
                                label_attrs = h5_data[label_number][seg].attrs.keys()
                                for attribute in label_attrs:
                                    data = h5_data[label_number][seg].attrs[attribute]
                                    h5_create[label_number][seg].attrs[attribute] = data
                elif key == 'map':
                    map_path = h5_create[key]
                    map_data = map_path.create_dataset('map',data=map)
                    for attribute in map_attrs:
                        data = h5_data[key]['map'].attrs[attribute]
                        map_data.attrs[attribute] = data
                elif key == 'tf_static':             
                    for tf_ in tf:
                        tf_static = os.path.join(key,tf_)
                        h5_tf = h5_create.create_group(tf_static)
                        tf_attrs = h5_data[key][tf_].attrs.keys()
                        for attribute in tf_attrs:
                            data = h5_data[key][tf_].attrs[attribute]
                            h5_tf.attrs[attribute] = data
                        tf_path = h5_create[tf_static]
                        read_tf_key = h5_data[tf_static].keys()
                        for tf_key in read_tf_key:
                            tf_data = h5_data[tf_static][tf_key][()]
                            h5_tf_dataset = tf_path.create_dataset(tf_key,data=tf_data)

                            tf_attrs = h5_data[key][tf_][tf_key].attrs.keys()
                            for attribute in tf_attrs:
                                data = h5_data[key][tf_][tf_key].attrs[attribute]
                                h5_tf_dataset.attrs[attribute] = data

        with h5py.File(val_file,'w') as h5_create:  
            for key in h5_main:
                h5_create.create_group(key)
                if key == 'data':
                    num=0
                    for i in track(val_list):
                        image_tag = h5_data[key][str(i)].keys()
                        h5_create.create_group(f'{key}/{num}')
                        data_path = h5_create[key][str(num)]
                        num+=1
                        for tag in image_tag:
                            if tag == 'oxts' or tag == 'world_to_pose':
                                img_group = data_path.create_group(tag)
                                img_keys = h5_data[key][str(i)][tag].keys()
                                img_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in img_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_group.attrs[attribute] = data
                                for img_name in img_keys:
                                    image = h5_data[key][str(i)][tag][img_name][()]
                                    img_dataset = data_path[tag].create_dataset(img_name,data=image)
                                    
                                    img_attrs = h5_data[key][str(i)][tag][img_name].attrs.keys()
                                    for attribute in img_attrs:
                                        data = h5_data[key][str(i)][tag][img_name].attrs[attribute]
                                        img_dataset.attrs[attribute] = data
                            elif tag == 'odometry':
                                continue
                            else:
                                image = h5_data[key][str(i)][tag][()]
                                img_dataset = data_path.create_dataset(tag,data=image)
                                image_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in image_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_dataset.attrs[attribute] = data
                if key == 'header':
                    h5_header = h5_create[key]
                    h5_header.create_dataset('length',data = length)
                elif key == 'intrinsic':            
                    for name in image_name:
                        intrinsic_name = os.path.join('intrinsic',name)
                        intrinsic_group = h5_create.create_group(intrinsic_name)
                        intrinsic_attrs = h5_data[intrinsic_name].attrs.keys()
                        for attribute in intrinsic_attrs:
                            data = h5_data[key][name].attrs[attribute]
                            intrinsic_group.attrs[attribute] = data
                        for intrinsic in intrinsic_key:
                            intrinsic_data = h5_data[key][name][intrinsic][()]
                            h5_intrinsic = h5_create[key][name]
                            h5_intrinsic.create_dataset(intrinsic,data=intrinsic_data)
                elif key == 'label':
                    h5_create.create_group('label/kitti360')
                    for name in label:
                        label_number = os.path.join(f'{key}/kitti360',name)
                        h5_create.create_group(label_number)
                        label_key = h5_data[label_number].keys()
                        label_path = h5_create[label_number]
                        for seg in label_key:
                            label_data = h5_data[label_number][seg][()]
                            label_path.create_dataset(seg,data=label_data)
                            if seg == 'color':
                                label_attrs = h5_data[label_number][seg].attrs.keys()
                                for attribute in label_attrs:
                                    data = h5_data[label_number][seg].attrs[attribute]
                                    h5_create[label_number][seg].attrs[attribute] = data
                elif key == 'map':
                    map_path = h5_create[key]
                    map_data = map_path.create_dataset('map',data=map)
                    for attribute in map_attrs:
                        data = h5_data[key]['map'].attrs[attribute]
                        map_data.attrs[attribute] = data
                elif key == 'tf_static':             
                    for tf_ in tf:
                        tf_static = os.path.join(key,tf_)
                        h5_tf = h5_create.create_group(tf_static)
                        tf_attrs = h5_data[key][tf_].attrs.keys()
                        for attribute in tf_attrs:
                            data = h5_data[key][tf_].attrs[attribute]
                            h5_tf.attrs[attribute] = data
                        tf_path = h5_create[tf_static]
                        read_tf_key = h5_data[tf_static].keys()
                        for tf_key in read_tf_key:
                            tf_data = h5_data[tf_static][tf_key][()]
                            h5_tf_dataset = tf_path.create_dataset(tf_key,data=tf_data)

                            tf_attrs = h5_data[key][tf_][tf_key].attrs.keys()
                            for attribute in tf_attrs:
                                data = h5_data[key][tf_][tf_key].attrs[attribute]
                                h5_tf_dataset.attrs[attribute] = data

        with h5py.File(test_file,'w') as h5_create:  
            for key in h5_main:
                h5_create.create_group(key)
                if key == 'data':
                    num=0
                    for i in track(values):
                        image_tag = h5_data[key][str(i)].keys()
                        h5_create.create_group(f'{key}/{num}')
                        data_path = h5_create[key][str(num)]
                        num+=1
                        for tag in image_tag:
                            if tag == 'oxts' or tag == 'world_to_pose':
                                img_group = data_path.create_group(tag)
                                img_keys = h5_data[key][str(i)][tag].keys()
                                img_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in img_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_group.attrs[attribute] = data
                                for img_name in img_keys:
                                    image = h5_data[key][str(i)][tag][img_name][()]
                                    img_dataset = data_path[tag].create_dataset(img_name,data=image)
                                    
                                    img_attrs = h5_data[key][str(i)][tag][img_name].attrs.keys()
                                    for attribute in img_attrs:
                                        data = h5_data[key][str(i)][tag][img_name].attrs[attribute]
                                        img_dataset.attrs[attribute] = data
                            elif tag == 'odometry':
                                continue
                            else:
                                image = h5_data[key][str(i)][tag][()]
                                img_dataset = data_path.create_dataset(tag,data=image)
                                image_attrs = h5_data[key][str(i)][tag].attrs.keys()
                                for attribute in image_attrs:
                                    data = h5_data[key][str(i)][tag].attrs[attribute]
                                    img_dataset.attrs[attribute] = data
                if key == 'header':
                    h5_header = h5_create[key]
                    h5_header.create_dataset('length',data = length)
                elif key == 'intrinsic':            
                    for name in image_name:
                        intrinsic_name = os.path.join('intrinsic',name)
                        intrinsic_group = h5_create.create_group(intrinsic_name)
                        intrinsic_attrs = h5_data[intrinsic_name].attrs.keys()
                        for attribute in intrinsic_attrs:
                            data = h5_data[key][name].attrs[attribute]
                            intrinsic_group.attrs[attribute] = data
                        for intrinsic in intrinsic_key:
                            intrinsic_data = h5_data[key][name][intrinsic][()]
                            h5_intrinsic = h5_create[key][name]
                            h5_intrinsic.create_dataset(intrinsic,data=intrinsic_data)
                elif key == 'label':
                    h5_create.create_group('label/kitti360')
                    for name in label:
                        label_number = os.path.join(f'{key}/kitti360',name)
                        h5_create.create_group(label_number)
                        label_key = h5_data[label_number].keys()
                        label_path = h5_create[label_number]
                        for seg in label_key:
                            label_data = h5_data[label_number][seg][()]
                            label_path.create_dataset(seg,data=label_data)
                            if seg == 'color':
                                label_attrs = h5_data[label_number][seg].attrs.keys()
                                for attribute in label_attrs:
                                    data = h5_data[label_number][seg].attrs[attribute]
                                    h5_create[label_number][seg].attnum_elementsrs[attribute] = data
                elif key == 'map':
                    map_path = h5_create[key]
                    map_data = map_path.create_dataset('map',data=map)
                    for attribute in map_attrs:
                        data = h5_data[key]['map'].attrs[attribute]
                        map_data.attrs[attribute] = data
                elif key == 'tf_static':             
                    for tf_ in tf:
                        tf_static = os.path.join(key,tf_)
                        h5_tf = h5_create.create_group(tf_static)
                        tf_attrs = h5_data[key][tf_].attrs.keys()
                        for attribute in tf_attrs:
                            data = h5_data[key][tf_].attrs[attribute]
                            h5_tf.attrs[attribute] = data
                        tf_path = h5_create[tf_static]
                        read_tf_key = h5_data[tf_static].keys()
                        for tf_key in read_tf_key:
                            tf_data = h5_data[tf_static][tf_key][()]
                            h5_tf_dataset = tf_path.create_dataset(tf_key,data=tf_data)

                            tf_attrs = h5_data[key][tf_][tf_key].attrs.keys()
                            for attribute in tf_attrs:
                                data = h5_data[key][tf_][tf_key].attrs[attribute]
                                h5_tf_dataset.attrs[attribute] = data
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sequence', type=str,default=['00','02','03','04','05','06','07','09','10'],nargs='*',choices=['00','02','03','04','05','06','07','09','10'])
    parser.add_argument('-t','--train_frame',type=int,default=500,required=True)
    parser.add_argument('-v','--val_frame',type=int,default=50,required=True)
    args = parser.parse_args()
    for i in range(len(args.sequence)):
        file = f'./kitti_datasets/kitti360_seq{args.sequence[i]}.hdf5'
        os.makedirs(f'./kitti_datasets/mini_test/train', exist_ok=True)
        os.makedirs(f'./kitti_datasets/mini_test/val', exist_ok=True)
        os.makedirs(f'./kitti_datasets/mini_test/test', exist_ok=True)
        train_file = f'./kitti_datasets/mini_test/train/kitti360_seq{args.sequence[i]}_train.hdf5'
        val_file = f'./kitti_datasets/mini_test/val/kitti360_seq{args.sequence[i]}_val.hdf5'
        test_file = f'./kitti_datasets/mini_test/test/kitti360_seq{args.sequence[i]}_test.hdf5'

        hdf5_input(file, train_file, val_file, test_file, args.train_frame, args.val_frame)
        print(f'sequence{args.sequence[i]} complete')
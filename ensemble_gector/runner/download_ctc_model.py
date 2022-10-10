from data_downloader.downloader import Downloader
 
model_dir = 'miduCTC_v3.7.0_csc3model'



if __name__ == '__main__':
    Downloader(local_path='model/{}.zip'.format(model_dir),
               remote_path='/public/nlp/yjy-gen-corrector/{}.zip'.format(model_dir)).get_path()





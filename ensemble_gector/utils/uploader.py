import ftplib
import os
import sys
import zipfile



class Uploader(object):
    """上传模型到ftp server"""

    def __init__(self, local_path, remote_path, host='10.100.2.75'):
        """
        Args:
            local_path: str 本地路径，需要打包上传的路径
            remote_path: str 远程路径，end with ’.zip'
            host:
        """
        self._local_path = local_path
        self._remote_path = remote_path
        self._host = host

    def push_path(self, output_filename=None):
        if not output_filename:
            output_filename = self._local_path + '.zip'
        self._make_zip(source_dir=self._local_path,
                       output_filename=output_filename)  # 打包路径

        self._upload(output_filename)  # 上传zip

    def _upload(self, local_path):
        # path是file，不是路径
        sys.stdout.write(
            '\nstart to upload {} to ftp ...\n'.format(local_path))
        ftp = ftplib.FTP(self._host, user='yjy.public',
                         passwd='midu123.com', timeout=60)
        file = open(local_path, "rb")
        ftp.storbinary("STOR {}".strip().format(self._remote_path), file)
        sys.stdout.write('success, save {} to {}\n'.format(
            local_path, self._remote_path))

        file.close()
        ftp.quit()

    def _make_zip(self, source_dir, output_filename):
        """把source_dir目录下所有的文件打包成zip"""

        zipf = zipfile.ZipFile(output_filename, 'w')
        pre_len = len(os.path.dirname(source_dir))
        for parent, dirnames, filenames in os.walk(source_dir):
            for filename in filenames:
                pathfile = os.path.join(parent, filename)
                arcname = pathfile[pre_len:].strip(os.path.sep)  # 相对路径
                zipf.write(pathfile, arcname)
        zipf.close()
        sys.stdout.write('successfully zip file_path {} to {}'.format(
            source_dir, output_filename))


def run_upload(local_path, remote_path):
    print('local path:{}'.format(local_path))
    print('remote path:{}'.format(remote_path))
    Uploader(local_path, remote_path).push_path()

    print('end')


if __name__ == '__main__':

    # parse_args_and_run(run_upload)
    local_path = 'data/ctc_comp/new/preliminary_b_data'
    # remote_path可以打时间标签
    remote_path = '/public/nlp/yjy-gen-corrector/preliminary_b_data_0504.zip'
    Uploader(local_path, remote_path).push_path()
    print('end')

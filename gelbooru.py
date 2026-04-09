import requests
import time
import os
from lxml.html import fromstring
from urllib.parse import parse_qs, urlparse
from tqdm import tqdm
import subprocess

# --- 1. 核心爬虫类 (保持原有逻辑) ---
class GelbooruCrawler():
    def __init__(self, cookies: str):
        """基本的 Gelbooru 爬虫"""
        self.cookies = cookies
        self.cookie_dict = {}
        self.request_delay = 0.2 # 稍微增加一点延时以防封禁

        # 解析 Cookie
        if self.cookies:
            for item in [item.strip() for item in self.cookies.split(';') if item.strip()]:
                try:
                    k, v = item.split('=', 1)
                    self.cookie_dict[k] = v
                except ValueError:
                    continue

    def _get(self, url, **kwargs):
        headers = kwargs.get('headers', {})
        headers.update({'cookie': self.cookies})
        headers.update({
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'referer': 'https://gelbooru.com'
        })
        kwargs['headers'] = headers

        time.sleep(self.request_delay)
        kwargs['params'] = kwargs.get('params', {})

        # 注意：这里的 API Key 和 User ID 是硬编码的，建议后续改为参数传入
        kwargs['params'].update({
            'api_key': '8e37ecc9dc37b1fd9c9d6d1ad7de26ab585059fc62505b544f39d030c2cc42d1',
            'user_id': '582042',
        })

        return requests.get(url, **kwargs)

    def get_posts_info(self, qs: dict, page_begin=0, page_end=1):
        url = 'https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1'
        for page in range(page_begin, page_end):
            params = {'pid': page, **qs}
            try:
                resp = self._get(url, params=params)
                data = resp.json()
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                continue

            if 'post' in data:
                yield (page, data['post'])
            else:
                return # End of results

# --- 2. 封装函数：获取 URL 列表 ---
def fetch_gelbooru_data(
    cookie_str: str,
    tags_str: str,
    pages_to_crawl: int = 10,
    images_to_crawl: int = 100,
    begin_page: int = 0
) -> list[dict]:
    """
    接受原有参数，返回一个包含 {'url': str, 'id': int, 'tags': str} 的列表
    """
    crawler = GelbooruCrawler(cookie_str)
    
    # 处理多 tag (逗号分隔)
    tag_list = [tag.strip() for tag in tags_str.split(',')]
    
    collected_data = []
    total_images_count = 0
    
    print(f"Start crawling tags: {tag_list}")

    for tag in tag_list:
        # 如果总数已达标，停止处理下一个 tag
        if total_images_count >= images_to_crawl:
            break

        page_iterable = crawler.get_posts_info(
            {'tags': tag}, 
            page_begin=begin_page, 
            page_end=begin_page + pages_to_crawl
        )

        pbar = tqdm(page_iterable, desc=f"Crawling '{tag}'", unit="page")
        
        for page_num, posts in pbar:
            if not posts:
                break
                
            for item in posts:
                if total_images_count >= images_to_crawl:
                    break
                
                # 提取我们需要的数据
                file_url = item.get('file_url')
                post_id = item.get('id')
                post_tags = item.get('tags')

                if file_url and post_id:
                    collected_data.append({
                        'url': file_url,
                        'id': post_id,
                        'tags': post_tags
                    })
                    total_images_count += 1
            
            if total_images_count >= images_to_crawl:
                break
        
        pbar.close()

    print(f"Crawling finished. Collected {len(collected_data)} items.")
    return collected_data

# --- 3. 封装函数：下载到本地 ---
def download_images_locally(data_list: list[dict], save_dir: str = './GelbooruUrls'):
    """
    下载 fetch_gelbooru_data 返回的数据列表到指定文件夹
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    print(f"Downloading {len(data_list)} files to {save_dir}...")

    # 设置 User-Agent 避免下载时 403
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }

    for item in tqdm(data_list, desc="Downloading"):
        url = item['url']
        file_id = item['id']
        
        # 获取文件扩展名
        ext = os.path.splitext(url)[1]
        if not ext:
            ext = '.jpg' # 默认回退
            
        filename = f"{file_id}{ext}"
        file_path = os.path.join(save_dir, filename)

        # 如果文件已存在则跳过
        if os.path.exists(file_path):
            continue

        try:
            # 使用 stream=True 避免大文件占用内存
            with requests.get(url, headers=headers, stream=True, timeout=15) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    print(f"All downloads completed in {save_dir}")


def download_images_locally_by_aria2(data_list: list[dict], save_dir: str = './GelbooruUrls', cookie = ''):
    """
    使用 aria2c 下载 fetch_gelbooru_data 返回的数据列表。
    需要系统已安装 aria2c 并配置在 PATH 中。
    """
    if not data_list:
        print("No images to download.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 1. 创建 aria2 的输入文件 (input file)
    # 格式为:
    # URL
    #   out=FILENAME
    input_file_path = os.path.join(save_dir, 'aria2_input.txt')
    
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            url = item['url']
            file_id = item['id']
            
            # 确定扩展名
            ext = os.path.splitext(url)[1]
            if not ext:
                ext = '.jpg'
            
            filename = f"{file_id}{ext}"
            
            # 写入 aria2 格式: 第一行 URL，第二行 tab + out=文件名
            f.write(f"{url}\n")
            f.write(f"\tout={filename}\n")

    print(f"Generated aria2 input file at {input_file_path}")
    print(f"Starting aria2 download for {len(data_list)} files...")


    cmd = [
        'aria2c',
        '-i', input_file_path,
        '-d', save_dir,
        '-x', '16', 
        '-s', '16', 
        '-j', '16',
        '-c',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        '--all-proxy', 'http://127.0.0.1:10809',
        '--header', 'cookie: {cookie}',
        '--header', 'referer: https://gelbooru.com/',
    ]

    try:
        # 调用系统命令
        subprocess.run(cmd, check=True)
        print(f"Aria2 download completed in {save_dir}")
        
        # 3. 下载完成后删除临时输入文件
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
            
    except FileNotFoundError:
        print("Error: 'aria2c' command not found. Please install aria2 or add it to your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Aria2 encountered an error: {e}")

# --- 4. 执行部分 (Example Usage) ---

# 配置参数 (模拟 Colab 表单输入)
COOKIE_ENV = 'PHPSESSID=XGH9W6YK5d8TcA%2C5YTFekAlqOcK-nZjn0uTq89ByZJn05%2CvoRqwVglyeQ271a43-GDx1jOrmkdNtfWepBR54Kz01Qg8II2yZNnZvtMvW%2C3wXu38GBGfD4owPb6ZNsyDd; fringeBenefits=yup; comment_threshold=0; post_threshold=0; user_id=582042; pass_hash=cc072643dfecf5772e697e61563cd15ee5b5becd; cf_clearance=rmUen6tdycsIvWRwlmDABLX1Z2c1gJBsKy7c5KyLz6E-1768222789-1.2.1.1-uHEjDITrVrwePHN8SbfmYDlGsuAMtAxpxn9NL.6wiNAnOu.Pb..d3D7UL8E4GkH71dQHgxjalAfg8D59HsxIYffJe3tBizOZ0HEI_REsOd6rbstNs_XmR9MySyq6Zu9t4YXHcQBmADgkmwFJsjIA.VgOpWWhgM0Qi_u4PGco8UUnIHFJLCcqCRmv1BC7NZYM3N0rOL5Ssn.FMczJCIhdaejjjvEPH2vw6prSiiX4v_8; bnState_2099173=%7B%22impressions%22%3A1%2C%22delayStarted%22%3A0%7D; __PPU_cl_tl=zYAAgqFs0mj0bq2hYwYBgqFs0mmcYr2hY1g; bnState_2099169=%7B%22impressions%22%3A1%2C%22delayStarted%22%3A0%7D; UGVyc2lzdFN0b3JhZ2U=%7B%22CAIFRQ%22%3A%22ADU%252B1QAAAAAAAAAD%22%2C%22CAIFRT%22%3A%22ADU%252B1QAAAABpnoHQ%22%2C%22MTIFRQ%22%3A%22AEfd4AAAAAAAAAAD%22%2C%22MTIFRT%22%3A%22AEfd4AAAAABpnoHQ%22%7D; bnState_2099171=%7B%22impressions%22%3A1%2C%22delayStarted%22%3A0%7D'
MAX_PAGES = 100
MAX_IMAGES = 80

os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

NEG_TAGS = '-animated'

def main():
    tags = input('Tags: ').strip() + ' ' + NEG_TAGS
    save_dir = tags.split(' ')[0]

    print(f'Crawling {tags}, Output Dir:')
    print(f'{save_dir}')


    image_data_list = fetch_gelbooru_data(
        cookie_str=COOKIE_ENV,
        tags_str=tags,
        pages_to_crawl=MAX_PAGES,
        images_to_crawl=MAX_IMAGES
    )

    download_images_locally_by_aria2(image_data_list, save_dir=save_dir, cookie=COOKIE_ENV)


if __name__ == '__main__':
    main()
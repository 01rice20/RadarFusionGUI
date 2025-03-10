import socket
import csv
import pandas as pd
import time

def start_server(host='192.168.0.200', port=7, csv_file='./output.csv'):
    # 建立 socket 物件
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 設定 socket 參數
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 綁定地址和端口
    server_socket.bind((host, port))
    # 開始監聽
    server_socket.listen(5)
    print(f'Server started and listening on {host}:{port}')

    try:
        # 接受客戶端連接
        client_socket, addr = server_socket.accept()
        print(f'Accepted connection from {addr}')
        # 傳送CSV檔中的數據給客戶端
        send_csv_data(client_socket, csv_file)
    except KeyboardInterrupt:
        print('Server is shutting down.')
    finally:
        server_socket.close()

# def send_csv_data(client_socket, csv_file):
#     try:
#         reader = csv.reader(csv_file)
#         while 1 :
#             test = reader[-1]
#             reader = csv.reader(csv_file)
#             # 將每一行轉換成逗號分隔的字串
#             for row in reader[-1]:
#                 data = ','.join(row) + '\n'
#                 print(data)
#                 # 傳送給客戶端
#                 if(reader.tail(1) != test):
#                     client_socket.sendall(data.encode())
#     except Exception as e:
#         print(f'Error reading CSV file: {e}')
#     finally:
#         client_socket.close()

def send_csv_data(client_socket, csv_file):
    data = []
    
    for i in range(60):
        with open('./output.csv', 'r') as f:
            [next(f) for _ in range(i)]
            reader = next(f)
            print(reader)
            client_socket.sendall(str(reader).encode())
        time.sleep(1.5)
    
    client_socket.close()

if __name__ == '__main__':
    start_server()

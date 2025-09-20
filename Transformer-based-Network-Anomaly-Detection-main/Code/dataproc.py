import pandas as pd
import glob
import sys

try:
    # 获取当前目录下所有ISCX CSV文件
    csv_files = glob.glob('*_ISCX.csv')
    print(f"Found {len(csv_files)} CSV files to process")

    # 读取并连接所有CSV文件
    dfs = []
    for file in csv_files:
        try:
            print(f"Reading {file}...")
            df = pd.read_csv(file, encoding='utf-8')  # 使用latin1编码读取PCAP CSV文件
            dfs.append(df)
            print(f"Successfully read {len(df)} rows from {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}", file=sys.stderr)

    if not dfs:
        print("No valid data frames to concatenate", file=sys.stderr)
        sys.exit(1)

    # 垂直连接所有DataFrame
    combined_df = pd.concat(dfs, axis=0)
    print(f"Successfully combined to {len(combined_df)} total rows")

    # 保存合并后的文件
    combined_df.to_csv('combined_traffic_data.csv', index=False)
    print("Results saved to combined_traffic_data.csv")
    
    # 提取最后一列并保存
    last_column = combined_df.iloc[:, -1]
    last_column.to_csv('last_column.csv', index=False, header=True)
    print(f"Last column '{combined_df.columns[-1]}' saved to last_column.csv")

except Exception as e:
    print(f"Script failed: {str(e)}", file=sys.stderr)
    sys.exit(1)

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime

st.set_page_config(layout="wide")
def calculate_reachability_with_and(unit_matrix):
    rea_matrix = unit_matrix.copy()
    matrix_list = [rea_matrix.copy()]

    while True:
        rea_next_power = np.dot(rea_matrix, unit_matrix) > 0  # 現在の可到達行列と初期の行列のAND演算
        rea_next_power = rea_next_power.astype(int)
        
        if np.array_equal(rea_matrix, rea_next_power):  # 可到達行列が変化しなくなったら終了
            break
        
        matrix_list.append(rea_next_power.copy())
        rea_matrix = rea_next_power.copy()

    return  rea_matrix,matrix_list


def data_format(df_csv):
    #matrixフォーマット
    df_csv.fillna(0, inplace=True)
    df_matrix = df_csv.copy()
    df_matrix =df_matrix.iloc[:, 3:]
    new_header = df_matrix.iloc[0] 
    df_matrix = df_matrix[1:] 
    df_matrix.columns = new_header.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df_matrix = df_matrix.set_index(df_matrix.columns[0])

    #node_listフォーマット
    df_nodelist = df_csv.copy()
    df_nodelist = df_nodelist.iloc[:, :2]
    #これはデータ依存なので，ちゃんと適切な位置で切らないとだめ
    df_nodelist = df_nodelist.iloc[:-1]

    df_nodelist = df_nodelist.set_index(df_nodelist.columns[0])

    return df_matrix, df_nodelist


def main():
    st.title('ISM教材構造化法')
    st.write("以下のリンクからcsvファイルのテンプレートをダウンロードしてください．")
    link = '[csv_template_link](https://drive.google.com/drive/folders/1NquS1HCFg6mUTFNdkh9jcVnAQ27WakAr?usp=sharing)'
    st.markdown(link, unsafe_allow_html=True) 

    # ユーザーからの行列入力
    uploaded_file = st.file_uploader("Please select a CSV file", type=["csv"])

    if uploaded_file is not None:
            df_csv = pd.read_csv(uploaded_file)
            
            df_matrix,df_nodelist = data_format(df_csv)
            st.write('ノードリスト')
            st.dataframe(df_nodelist)            

            n = df_matrix.shape[1]
            n = int(n)

            identity_matrix = np.eye(n, dtype=int)
            df_matrix[identity_matrix == 1] = 1
            #df_matrix[:] = identity_matrix
            
            df = pd.DataFrame(np.zeros((n, n), dtype=int), columns=range(1, n+1), index=range(1, n+1))
            df[:] = df_matrix.values
            st.write('隣接行列+単位マトリクス')
            st.dataframe(df)

            adjacency_matrix = df.copy()
            adjacency_matrix[identity_matrix == 1] = 0
            
            #計算用マトリクス
            unit_matrix = df.values

            
            
            # 可到達行列の計算と結果表示
            if st.button('可到達行列の計算'):
                rea_matrix,matrix_list = calculate_reachability_with_and(unit_matrix)
                
                # 各累乗の行列を表示
                with st.expander("累乗結果の表示"):
                    for idx, matrix in enumerate(matrix_list, start=1):
                        st.write(f"{idx}乗の結果:")
                        st.dataframe(pd.DataFrame(matrix, columns=df.columns, index=df.index))
                        reachability = idx 

                col1, col2 = st.columns(2)
                with col1:
                    with st.container():
                        st.write("隣接行列+単位マトリクス")
                        st.dataframe(df)
                        st.write("")
                        st.write("")
                        
                with col2:
                    st.write(f"可到達行列({reachability}乗={reachability+1}乗)")
                    st.dataframe(pd.DataFrame(rea_matrix, columns=df.columns, index=df.index))
                    df_rea = pd.DataFrame(rea_matrix, columns=df.columns, index=df.index)
                    df_rea["SUM"] = df_rea.sum(axis=1)
                    unique_counts = df_rea['SUM'].nunique()
                    st.write(f"ノードの階層数：{unique_counts}階層（ノードの階層）")
                    st.write("")
                    st.write("")

                # ネットワークグラフ
                G = nx.DiGraph()
                for i in range(1, n + 1):
                    G.add_node(str(i))

                for i in adjacency_matrix.index:
                    for j in adjacency_matrix.columns[adjacency_matrix.loc[i] == 1]:
                        G.add_edge(str(i), str(j))

                pos_dict = {}
                levels = df_rea['SUM'].unique().tolist()
                levels.sort()

                # 各レベルでのノードを保持
                level_nodes = {level: [] for level in levels}
                for node, row in df_rea.iterrows():
                    level_nodes[row['SUM']].append(str(node))

                # 2つ以上上のノードにリンクするノードの特定
                upward_links = {}
                for node in G.nodes():
                    node_level = df_rea.at[int(node),'SUM']
                    for target in G.successors(node):
                        target_level = df_rea.at[int(target),'SUM']
                        if target_level > node_level + 1:
                            if node not in upward_links:
                                upward_links[node] = []
                            upward_links[node].append(target)

                # X軸位置の調整
                x_positions = {level: 0 for level in levels}  # 各レベルでの現在のX軸位置
                for level in levels:
                    nodes = level_nodes[level]
                    for node in nodes:
                        if node in upward_links:
                            # 2つ以上上のノードにリンクするノードはX軸を調整
                            pos_dict[node] = (x_positions[level] + 1, -level)
                            x_positions[level] += 2  # 調整した分だけX軸位置を変更
                        else:
                            # それ以外のノードは現在のX軸位置に配置
                            pos_dict[node] = (x_positions[level], -level)
                            x_positions[level] += 1  

                # 既に使われている位置を避ける
                occupied_positions = set(pos_dict.values())
                for node, pos in pos_dict.items():
                    while pos in occupied_positions:
                        pos = (pos[0] + 1, pos[1])
                        pos_dict[node] = pos
                    occupied_positions.add(pos)

                # グラフの描画
                fig, ax = plt.subplots()
                nx.draw(G, pos_dict, with_labels=True, arrows=True, ax=ax)
                
                # リンクの上にラベルを表示（オンオフ切り替えが出来ない，仕様？）
                labels = {(str(u), str(v)): f"{u}→{v}" for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=labels)
                
                col1, col2 = st.columns(2)
                with col1:
                    with st.container():
                        st.write('構造モデル')
                        st.pyplot(fig)
                        
                with col2:
                    st.write("Node_list")
                    st.dataframe(df_nodelist)

                #Node Listのhtmlテーブルの生成
                html_nodelist = '<table border="1"><thead><tr><th>Node:No.</th>'
                # ヘッダー行追加
                for col in df_nodelist.columns:
                    html_nodelist += f'<th>{col}</th>'
                html_nodelist += '</tr></thead><tbody>'
                # 行データ追加
                for index, row in df_nodelist.iterrows():
                    index_as_int = int(index)
                    html_nodelist += f'<tr><td>{index_as_int}</td>'
                    for col in df_nodelist.columns:
                        html_nodelist += f'<td>{row[col]}</td>'
                    html_nodelist += '</tr>'
                html_nodelist += '</tbody></table>'

                #隣接行列のhtmlテーブルの生成
                html_df = '<table border="1"><thead><tr><th></th>'
                #ヘッダー追加
                for col in df.columns:
                    html_df += f'<th>{col}</th>'
                html_df += '</tr></thead><tbody>'
                for index, row in df.iterrows():
                    index_as_int = int(index)
                    html_df += f'<tr><td>{index_as_int}</td>'
                    for col in df.columns:
                        html_df += f'<td>{row[col]}</td>'
                    html_df += '</tr>'
                html_df += '</tbody></table>'
                
                #可到達のhtmlテーブルの生成
                rea_matrix_df = pd.DataFrame(rea_matrix, columns=df.columns, index=df.index)
                html_rea_matrix = '<table border="1"><thead><tr><th></th>'
                # ヘッダー行の追加
                for col in rea_matrix_df.columns:
                    html_rea_matrix += f'<th>{col}</th>'  # ここを修正
                html_rea_matrix += '</tr></thead><tbody>'

                for index, row in rea_matrix_df.iterrows():
                    index_as_int = int(index)  # インデックスを整数にキャスト
                    html_rea_matrix += f'<tr><td>{index_as_int}</td>'
                    for col in rea_matrix_df.columns:
                        html_rea_matrix += f'<td>{row[col]}</td>'
                    html_rea_matrix += '</tr>'
                html_rea_matrix += '</tbody></table>'

                now = datetime.now()
                formatted_date = now.strftime("%Y/%m/%d %H:%M")
                
           
               
                #htmlを定義
                buf = BytesIO()
                fig.savefig(buf, format="png")
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                html_fig = f'<img src="data:image/png;base64,{data}" />'
                
                html_content = f"""
                <html>
                <head>
                <title>Data Report</title>
                </head>
                <body>
                <h1>Data Report</h1>
                <p>作成日: {formatted_date}</p>
                <p>隣接行列＋単位行列</p>
                {html_df}
                
                <p>可到達行列</p>
                {html_rea_matrix}
                <p>ノードリスト</p>
                {html_nodelist}
                <p>全体構造</p>
                {html_fig}
                </body>
                </html>
                """

                st.download_button(
                label="Download Report",
                data=html_content,
                file_name="Your_Report.html",
                mime="text/html"
                )
                

if __name__ == '__main__':
    main()

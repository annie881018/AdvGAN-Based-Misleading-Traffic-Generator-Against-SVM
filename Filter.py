def filtering(df):
    # max iat > min iat
    df_filtered = df[df['max iat'] >= df['min iat']]
    # max pkt len > min pkt len
    df_filtered = df_filtered[df_filtered['max pkt_length'] >=  df_filtered['min pkt_length']]
    # pkt count > 0
    df_filtered = df_filtered[df_filtered['packet count'] >= df_filtered['fin count']]
    df_filtered = df_filtered[df_filtered['packet count'] >= df_filtered['syn count']]
    df_filtered = df_filtered[df_filtered['packet count'] >= df_filtered['psh count']]
    df_filtered = df_filtered[df_filtered['packet count'] >= df_filtered['ack count']]


    # [FIN, ACK]
    df_filtered = df_filtered[df_filtered['fin count'] <= df_filtered['ack count']]
    # [PSH, ACK]
    df_filtered = df_filtered[df_filtered['psh count'] <= df_filtered['ack count']]

    # 新增條件 1: 若 pkt count > 3，則 flow duration >= 1
    df_filtered = df_filtered[~((df_filtered['packet count'] > 3) & (df_filtered['flow duration'] < 1))]

    # 新增條件 2: 若 flow duration > 0，則 pkt count > 0
    df_filtered = df_filtered[~((df_filtered['flow duration'] >= 1) & (df_filtered['packet count'] <= 3))]

    # 新增條件 3: flow duration >= max iat
    df_filtered = df_filtered[df_filtered['flow duration'] >= df_filtered['max iat']]

#     # 新增條件 1: 若 pkt count > 0，則 flow duration > 0
#     df_filtered = df_filtered[~((df_filtered['max iat'] > 0) & (df_filtered['flow duration'] <= 0))]

#     # 新增條件 2: 若 flow duration > 0，則 pkt count > 0
#     df_filtered = df_filtered[~((df_filtered['flow duration'] > 0) & (df_filtered['max iat'] <= 0))]


    # 新增條件 4: 所有 flags 加總 <= packet count
    df_filtered = df_filtered[
        (df_filtered['fin count'] + df_filtered['syn count'] +
         df_filtered['psh count'] + df_filtered['ack count']) <= (df_filtered['packet count'] * 2)
    ]
    df_filtered = df_filtered[
        (df_filtered['fin count'] + df_filtered['syn count'] +
         df_filtered['psh count'] + df_filtered['ack count']) >= (df_filtered['packet count'] * 0.5)
    ]


    df_filtered = df_filtered[~(
    (df_filtered['ack count'] > 5) &
    ((df_filtered['fin count'] + df_filtered['syn count'] + df_filtered['psh count']) < (df_filtered['ack count'] * 0.25))
    )]
    df_filtered = df_filtered[
        (df_filtered['max iat']) >= (df_filtered['packet count'] * 0.4)
    ]
#     df_filtered = df_filtered[
#         (df_filtered['flow duration']) >= (df_filtered['packet count'] * 0.4)
#     ]
#     df_filtered = df_filtered[
#         (df_filtered['flow duration']) <= (df_filtered['packet count'] * 16654)
#     ]
    df_filtered = df_filtered[
        (df_filtered['max iat']) >= (df_filtered['flow duration'] / (df_filtered['packet count']-1))
    ]

#     df_filtered = df_filtered[~((df_filtered['packet count'] > 0) & (df_filtered['fin count'] + df_filtered['syn count'] +
#          df_filtered['psh count'] + df_filtered['ack count'] <= 0))]
#     df_filtered = df_filtered[~((df_filtered['packet count'] <= 0) & (df_filtered['fin count'] + df_filtered['syn count'] +
#          df_filtered['psh count'] + df_filtered['ack count'] > 0))]


    return df_filtered
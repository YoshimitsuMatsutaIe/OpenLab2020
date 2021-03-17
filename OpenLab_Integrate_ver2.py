'''
オープンラボ2020の運動学プログラム（仮2）
'''

# モジュール読み込み

import numpy as np
import math
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pyfirmata
import cv2
import time
aruco = cv2.aruco
p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


# 関数宣言

def inv_kinema_cal_3(JOINT_ANGLE_OFFSET, L, H, position_to_move):
    """逆運動学を解析的に解く関数.

    指先のなす角がηになるようなジョイント角度拘束条件を追加して逆運動学問題を解析的に解く

    引数1：リンク長さの配列．nd.array(6)．単位は[m]
    引数2：リンク高さの配列．nd.array(1)．単位は[m]
    引数3：目標位置（直交座標系）行列．nd.array((3, 1))．単位は[m]

    戻り値（成功したとき）：ジョイント角度配列．nd.array((6))．単位は[°]
    戻り値（失敗したとき）：引数に関係なくジョイント角度配列（90, 90, 90, 90, 90, 0）．nd.array((6))．単位は[°]を返す
    ※戻り値のq_3,q_4はサーボの定義と異なる
    """
    
    final_offset = 0.012
    #final_offset = 0

    # position_to_move（移動先位置）の円筒座標系表現
    r_before = math.sqrt(position_to_move[0, 0] ** 2 + position_to_move[1, 0] ** 2) + 0.03
    r_to_move = math.sqrt(r_before ** 2 + final_offset ** 2)  # [m]
    #r_to_move = math.sqrt(r_before ** 2)  # [m]
    #theta_to_move = np.arctan2(position_to_move[1, 0], position_to_move[0, 0]) # [rad]
    theta_to_move = np.arctan2(position_to_move[1, 0], position_to_move[0, 0]) - np.arcsin(final_offset / r_before) # [rad]
    #theta_to_move = np.arccos(position_to_move[0, 0] / r_to_move) - np.arcsin(final_offset / r_before) # [rad]
    z_to_move = position_to_move[2, 0]  # [m]
    print('移動先の円筒座標系表現は\n', r_to_move, '[m]\n', int(theta_to_move * 180 / np.pi), '[°]\n', z_to_move, '[m]')
    
    # 計算のため定義する定数
    A = L[2]
    B = L[3]

    # 逆運動学解析解計算

    #old1 = time.time()

    deta = np.pi / 180  # ηの刻み幅．i[°]ずつ実行

    eta = np.arange(0, np.pi + deta, deta, dtype = 'float64')  # 全ηの配列
    print('etaの形は', eta.shape)

    # パターンa
    q_2_a = np.arcsin((A ** 2 - B ** 2 + (r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2) \
        / (2 * A * np.sqrt((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2))) \
        - np.arctan((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) / (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)))  # [rad]

    qlist_a_1 = np.concatenate([[eta], [q_2_a]], 0)  # 縦に連結
    qlist_a_2 = np.delete(qlist_a_1, np.where((np.isnan(qlist_a_1)) | (qlist_a_1 < 0) | ((np.pi * (1 - JOINT_ANGLE_OFFSET[1] / 180)) < qlist_a_1))[1], 1)  # q_2_aがNAN，またはジョイント制限外の列を削除

    q_3_a = np.arcsin((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(qlist_a_2[0, :]) - H[0] * np.sin(qlist_a_2[0, :])- A * np.cos(qlist_a_2[1, :]) + z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(qlist_a_2[0, :]) - H[0] * np.cos(qlist_a_2[0, :]) - A * np.sin(qlist_a_2[1, :])) \
        / (np.sqrt(2) * B)) - qlist_a_2[1, :] + np.pi / 4  # [rad]

    qlist_a_3 = np.concatenate([qlist_a_2, [q_3_a]], 0)  # 縦に連結
    qlist_a_4 = np.delete(qlist_a_3, np.where((np.isnan(qlist_a_3)) | (qlist_a_3 < (np.pi * (JOINT_ANGLE_OFFSET[2] / 180))) | (np.pi < qlist_a_3))[1], 1)  # q_3_aがNAN，またはジョイント制限外の列を削除

    q_4_a = -qlist_a_4[0, :] + np.pi - qlist_a_4[1, :] - qlist_a_4[2, :]

    qlist_a_5 = np.concatenate([qlist_a_4, [q_4_a]], 0)  # 縦に連結
    qlist_a_6 = np.delete(qlist_a_5, np.where((qlist_a_5 < (np.pi * (JOINT_ANGLE_OFFSET[3] / 180))) | (np.pi < qlist_a_5))[1], 1)  # q_4_aがジョイント制限外の列を削除
    #print('qlist_a_6の形は', qlist_a_6.shape)
    #print('qlist_a_6 = ', qlist_a_6)


    # パターンb
    q_2_b = np.arcsin((A ** 2 - B ** 2 + (r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2) \
        / (2 * A * np.sqrt((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2))) \
        - np.arctan((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) / (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)))  # [rad]

    qlist_b_1 = np.concatenate([[eta], [q_2_b]], 0)  # 縦に連結
    qlist_b_2 = np.delete(qlist_b_1, np.where((np.isnan(qlist_b_1)) | (qlist_b_1 < 0) | ((np.pi * (1 - JOINT_ANGLE_OFFSET[1] / 180))< qlist_a_1))[1], 1)  # q_2_bがNAN，またはジョイント制限外の列を削除

    q_3_b = np.pi - np.arcsin((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(qlist_b_2[0, :]) - H[0] * np.sin(qlist_b_2[0, :])- A * np.cos(qlist_b_2[1, :]) + z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(qlist_b_2[0, :]) - H[0] * np.cos(qlist_b_2[0, :]) - A * np.sin(qlist_b_2[1, :])) \
        / (np.sqrt(2) * B)) - qlist_b_2[1, :] + np.pi / 4  # [rad]

    qlist_b_3 = np.concatenate([qlist_b_2, [q_3_b]], 0)  # 縦に連結
    qlist_b_4 = np.delete(qlist_b_3, np.where((np.isnan(qlist_b_3)) | (qlist_b_3 < (np.pi * (JOINT_ANGLE_OFFSET[2] / 180))) | (np.pi < qlist_b_3))[1], 1)  # q_3_bがNAN，またはジョイント制限外の列を削除

    q_4_b = -qlist_b_4[0, :] + np.pi - qlist_b_4[1, :] - qlist_b_4[2, :]

    qlist_b_5 = np.concatenate([qlist_b_4, [q_4_b]], 0)  # 縦に連結
    qlist_b_6 = np.delete(qlist_b_5, np.where((qlist_b_5 < (np.pi * (JOINT_ANGLE_OFFSET[3] / 180))) | (np.pi < qlist_b_5))[1], 1)  # q_3_bがジョイント制限外の列を削除
    #print('qlist_b_6の形は', qlist_b_6.shape)
    #print('qlist_b_6 = ', qlist_b_6)


    # パターンc
    q_2_c = np.pi - np.arcsin((A ** 2 - B ** 2 + (r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2) \
        / (2 * A * np.sqrt((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2))) \
        - np.arctan((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) / (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)))  # [rad]

    qlist_c_1 = np.concatenate([[eta], [q_2_c]], 0)  # 縦に連結
    qlist_c_2 = np.delete(qlist_c_1, np.where((np.isnan(qlist_c_1)) | (qlist_c_1 < 0) | ((np.pi * (1 - JOINT_ANGLE_OFFSET[1] / 180))< qlist_a_1))[1], 1)  # q_2_cがNAN，またはジョイント制限外の列を削除

    q_3_c = np.arcsin((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(qlist_c_2[0, :]) - H[0] * np.sin(qlist_c_2[0, :])- A * np.cos(qlist_c_2[1, :]) + z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(qlist_c_2[0, :]) - H[0] * np.cos(qlist_c_2[0, :]) - A * np.sin(qlist_c_2[1, :])) \
        / (np.sqrt(2) * B)) - qlist_c_2[1, :] + np.pi / 4  # [rad]

    qlist_c_3 = np.concatenate([qlist_c_2, [q_3_c]], 0)  # 縦に連結
    qlist_c_4 = np.delete(qlist_c_3, np.where((np.isnan(qlist_c_3)) | (qlist_c_3 < (np.pi * (JOINT_ANGLE_OFFSET[2] / 180))) | (np.pi < qlist_c_3))[1], 1)  # q_3_cがNAN，またはジョイント制限外の列を削除

    q_4_c = -qlist_c_4[0, :] + np.pi - qlist_c_4[1, :] - qlist_c_4[2, :]

    qlist_c_5 = np.concatenate([qlist_c_4, [q_4_c]], 0)  # 縦に連結
    qlist_c_6 = np.delete(qlist_c_5, np.where((qlist_c_5 < (np.pi * (JOINT_ANGLE_OFFSET[3] / 180))) | (np.pi < qlist_c_5))[1], 1)  # q_3_cがジョイント制限外の列を削除
    #print('qlist_c_6の形は', qlist_c_6.shape)
    #print('qlist_c_6 = ', (qlist_c_6 * 180 / np.pi).astype('int64'))


    # パターンd
    q_2_d = np.pi - np.arcsin((A ** 2 - B ** 2 + (r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2) \
        / (2 * A * np.sqrt((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) ** 2 + (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)) ** 2))) \
        - np.arctan((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(eta) - H[0] * np.sin(eta)) / (z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(eta) - H[0] * np.cos(eta)))  # [rad]

    qlist_d_1 = np.concatenate([[eta], [q_2_d]], 0)  # 縦に連結
    qlist_d_2 = np.delete(qlist_d_1, np.where((np.isnan(qlist_d_1)) | (qlist_d_1 < 0) | ((np.pi * (1 - JOINT_ANGLE_OFFSET[1] / 180))< qlist_a_1))[1], 1)  # q_2_dがNAN，またはジョイント制限外の列を削除

    q_3_d = np.pi - np.arcsin((r_to_move  - (L[4] + L[5] + L[6]) * np.cos(qlist_d_2[0, :]) - H[0] * np.sin(qlist_d_2[0, :])- A * np.cos(qlist_d_2[1, :]) + z_to_move - L[0] - L[1] + (L[4] + L[5] + L[6]) * np.sin(qlist_d_2[0, :]) - H[0] * np.cos(qlist_d_2[0, :]) - A * np.sin(qlist_d_2[1, :])) \
        / (np.sqrt(2) * B)) - qlist_d_2[1, :] + np.pi / 4  # [rad]

    qlist_d_3 = np.concatenate([qlist_d_2, [q_3_d]], 0)  # 縦に連結
    qlist_d_4 = np.delete(qlist_d_3, np.where((np.isnan(qlist_d_3)) | (qlist_d_3 < (np.pi * (JOINT_ANGLE_OFFSET[2] / 180))) | (np.pi < qlist_d_3))[1], 1)  # q_3_dがNAN，またはジョイント制限外の列を削除

    q_4_d = -qlist_d_4[0, :] + np.pi - qlist_d_4[1, :] - qlist_d_4[2, :]

    qlist_d_5 = np.concatenate([qlist_d_4, [q_4_d]], 0)  # 縦に連結
    qlist_d_6 = np.delete(qlist_d_5, np.where((qlist_d_5 < (np.pi * (JOINT_ANGLE_OFFSET[3] / 180))) | (np.pi < qlist_d_5))[1], 1)  # q_3_dがジョイント制限外の列を削除
    #print('qlist_d_6の形は', qlist_d_6.shape)
    #print('qlist_d_6 = ', qlist_d_6)

    #print('ベクトル化で計算', time.time() - old1,'[s]')


    qlist_abcd_6 = np.concatenate([qlist_a_6, qlist_b_6, qlist_c_6, qlist_d_6], 1)  # パターンa,b,c,dの実行結果を横に連結
    print(qlist_abcd_6)

    qlist_q2norm = np.abs(np.pi / 2 - qlist_abcd_6[1, :])  # π/2 - q_2の絶対値
    print(qlist_q2norm)

    qlist_abcd_62 = np.concatenate([qlist_abcd_6, [qlist_q2norm]], 0)  # 縦連結
    print(qlist_abcd_62)

    k = np.where(qlist_abcd_62[4, :] == np.min(qlist_abcd_62[4, :]))  # 最もq_2がπ/2に近い列のタプルを取得

    print(k)
    print(qlist_abcd_62[:, k])

   
    # サーボ指令角度への変換とint化（pyFirmataのpwmは整数値指令しか受け付けない）
    q_1_command = int(np.round(theta_to_move * 180 / np.pi))  # [°]
    q_2_command = int(np.round(qlist_abcd_62[1, k] * 180 / np.pi))  # [°]
    q_3_command = int(np.round(qlist_abcd_62[2, k] * 180 / np.pi))  # [°]
    q_4_command = int(np.round(qlist_abcd_62[3, k] * 180 / np.pi))  # [°]
    q_5_command = int(np.round(np.pi / 2 * 180 / np.pi))  # [°]
    q_6_command = int(np.round(0 * 180 / np.pi))  # [°]

    z = np.array([q_1_command, q_2_command, q_3_command, q_4_command, q_5_command, q_6_command])
    print(z)
    return z



def initial_configration(JOINT_ANGLE_OFFSET, joint_angle_all):
    """マニピュレータを初期姿勢に戻す
    
    引数1：ジョイント角度のオフセットを格納したベクトル．nd.array((6))．単位は[°]
    引数2：サーボ変数（pyFirmata）を格納したベクトル．nd.array((6))．指定する際の単位は[°]

    戻り値：なし
    """

    print('初期姿勢に変更中...')
    joint_angle_all_initial = np.array([90, 135, 180, 180, 90, 90]) + JOINT_ANGLE_OFFSET
    print('ジョイント角の目標値は', joint_angle_all_initial)

    dt = 0.015  # ディレイ

    joint_angle_all_now = np.array([0, 0, 0, 0, 0, 0])
    for i in range(0, 6):  # 現在のジョイント角を読み込む
        joint_angle_all_now[i] = joint_angle_all[i].read() 
    
    print('ジョイント角の現在値は', joint_angle_all_now)
    #joint_angle_all_now = joint_angle_all_now + np.array([-5, 3, 3, 5, 4, 5])
    for i in range(0, 6):  # q_1~q_6を動かす
        print('ジョイント', i + 1, 'を変更中')
        if joint_angle_all_now[i] < joint_angle_all_initial[i]:  # 現在のジョイント角が指令角より小さいとき
            qseq = np.arange(joint_angle_all_now[i], joint_angle_all_initial[i] + 1, 1)
            print('現在のジョイント角が指令角より小さいときの指令角度数列', qseq)
            for j in qseq:  # 1°ずつ指令角に近づける
                joint_angle_all[i].write(j)
                #print('q', i + 1, 'の現在値は', joint_angle_all[i].read())
                time.sleep(dt)
        elif joint_angle_all_now[i] == joint_angle_all_initial[i]:  # 現在のジョイント角が指令角と等しいとき
            print('q', i + 1, 'の現在値は', joint_angle_all[i].read(), '変更なしでOK!')
            pass
        else:  # 現在のジョイント角が指令角より大きいとき
            qseq = np.arange(joint_angle_all_now[i], joint_angle_all_initial[i] - 1, -1)
            print('現在のジョイント角が指令角より大きいときの指令角度数列', qseq)
            for j in qseq:  # -1°ずつ指令角に近づける
                joint_angle_all[i].write(j)
                #print('q', i + 1, 'の現在値は', joint_angle_all[i].read())
                time.sleep(dt)

    print('初期姿勢に変更完了')


def change_configration(JOINT_ANGLE_OFFSET, position_to_move_before, joint_angle_all, z_offset):
    """マニピュレータの指先位置を目標位置のz_offset上に移動させる
    
    引数1：ジョイント角度のオフセットを格納したベクトル．nd.array((6))．単位は[°]
    引数2：目標位置（直交座標系）行列．nd.array((3, 1))．単位は[m]
    引数3：サーボ変数（pyFirmata）を格納したベクトル．nd.array((6))．指定するときの単位は[°]
    引数4：z_offset．単位は[m]

    戻り値：なし
    """

    position_to_move = position_to_move_before + np.array([[0], [0], [z_offset]])  # 移動目標をtarget_positionのz_offset上に設定
    q_to_move_before = inv_kinema_cal_3(JOINT_ANGLE_OFFSET, L, H, position_to_move)  # 逆運動学解析解を計算
    q_to_move = np.array([q_to_move_before[0], q_to_move_before[1], 180 - q_to_move_before[2], 180 - q_to_move_before[3], q_to_move_before[4], q_to_move_before[5]]) + \
        JOINT_ANGLE_OFFSET  # q_3,q_4を反転しoffsetを足す
    print('ジョイント角を', q_to_move, '[°]に変更中...')
    
    """
    # 直接実行
    joint_angle_all[0].write(q_to_move[0] + JOINT_ANGLE_OFFSET[0])
    joint_angle_all[1].write(q_to_move[1] + JOINT_ANGLE_OFFSET[1])
    joint_angle_all[2].write(q_to_move[2] + JOINT_ANGLE_OFFSET[2])
    joint_angle_all[3].write(q_to_move[3] + JOINT_ANGLE_OFFSET[3])
    #joint_angle_all[4].write(90 JOINT_ANGLE_OFFSET[4])
    """

    # 細かく実行
    dt = 0.025  # ディレイ

    joint_angle_1234_now = np.array([0, 0, 0, 0])  
    for i in range(0, 4):  # 現在のジョイント角を読み込む
        joint_angle_1234_now[i] = joint_angle_all[i].read()

    for i in np.array([0, 3, 1, 2]):  # q_1~q_4を動かす．順番は要検討
        print('ジョイント', i + 1, 'を変更中')
        if joint_angle_1234_now[i] < q_to_move[i]:  # 現在のジョイント角が指令角より小さいとき
            qseq = np.arange(joint_angle_1234_now[i], q_to_move[i] +1, 1)
            print('現在のジョイント角が指令角より小さいときの指令角度数列', qseq)
            for j in qseq:  # 1°ずつ指令角に近づける
                joint_angle_all[i].write(j)
                #print('q', i + 1, 'の現在値は', joint_angle_all[i].read())
                time.sleep(dt)
        elif joint_angle_1234_now[i] == q_to_move[i]:  # 現在のジョイント角が指令角と等しいとき
            print('q', i + 1, 'の現在値は', joint_angle_all[i].read(), '変更なしでOK!')
            pass
        else:  # 現在のジョイント角が指令角より大きいとき
            qseq = np.arange(joint_angle_1234_now[i], q_to_move[i] - 1, -1)
            print('現在のジョイント角が指令角より大きいときの指令角度数列', qseq)
            for j in qseq:  # -1°ずつ指令角に近づける
                joint_angle_all[i].write(j)
                #print('q', i + 1, 'の現在値は', joint_angle_all[i].read())
                time.sleep(dt)

    print('変更完了')
    #print('現在の姿勢は', coordinate_trans_from_cyl_to_car(for_kinema_cal_1(L, H, q_to_move)))


def detect_mark(n):
    """マーカーの位置座標x,yを検出する関数

    引数：マーカーの数．int型．

    戻り値：位置座標x,y．（2）配列．
    """

    n_mark = n+4
    X = np.empty((n_mark-4,2))
    m = np.empty((4,2))
    m2 = np.empty((n_mark-4,2))
    corners0 = [np.empty((1,4,2))]*n_mark
    corners1 = [np.empty((1,4,2))]*n_mark
    corners2 = [np.empty((1,4,2))]*(n_mark-4)
    corners3 = [np.empty((1,4,2))]*(n_mark-4)
    
    # VideoCaptureのインスタンスを作成する
    # 引数でカメラを選ぶ
    cap = cv2.VideoCapture(0)
    
    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
        
        windowsize = (640, 480)
        frame = cv2.resize(frame, windowsize)
    
        # 元画像を表示する
        #cv2.imshow('Raw Frame', frame)
    
        # 検出
        corners0, ids0, rejectedImgPoints0 = aruco.detectMarkers(frame, p_dict)
        
        # マーカーが検出できなかった場合はcontinue
        if len(ids0.ravel()) != n_mark:
            #print('検出1 マーカーの検出に失敗orマーカー個数不一致 設定したマーカーの個数:',n_mark,' 検出したマーカーの個数:',len(ids0.ravel()))
            continue
        
        for i0,c0 in zip(ids0.ravel(), corners0):
            corners1[i0] = c0.copy()
            
        m[0] = corners1[0][0][2]
        m[1] = corners1[1][0][3]
        m[2] = corners1[2][0][0]
        m[3] = corners1[3][0][1]
        
        # 変形後画像サイズ
        width, height = (800,405)
        marker_coordinates = np.float32(m)
        true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
        trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
        edframe = cv2.warpPerspective(frame,trans_mat,(width, height))
        
        # 検出
        corners2, ids2, rejectedImgPoints2 = aruco.detectMarkers(edframe, p_dict)
        
        # マーカーが検出できなかった場合はcontinue
        if len(ids2.ravel()) != n_mark-4:
            #print('検出2 マーカーの検出に失敗orマーカー個数不一致 設定したマーカーの個数:',n_mark-4,' 検出したマーカーの個数:',len(ids2.ravel()))
            continue
        else:
            break
        
    # 検出結果をオーバーレイ・表示
    #edframe2 = aruco.drawDetectedMarkers(edframe.copy(), corners2, ids2)
    #cv2.imshow('marker', edframe2)
    
    # 座標変換関数
    tr_x = lambda x : x - 400 # X軸 画像座標→実座標 
    tr_y = lambda y : 405 - y # Y軸 　〃
    
    # マーカー中心座標算出
    for i2,c2 in zip(ids2.ravel(), corners2):
        corners3[i2-4] = c2.copy()
        #print(corners3[i2-4])
        m2[i2-4] = corners3[i2-4][0].mean(axis=0)
        #print(m2[i2-4])
        X[i2-4][0]= round(tr_x(m2[i2-4][0]), 1)
        X[i2-4][1]= round(tr_y(m2[i2-4][1]), 1)
        
    '''
    # 座標算出・表示
    for k in range(n_mark-4):
        x = round(tr_x(m2[k][0]), 1)
        y = round(tr_y(m2[k][1]), 1)
        #print(f'■ マーカー {k+4} の中心位置 X={x}mm Y={y}mm ',end='')
        
    #print('')
    
    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break
    '''
    
    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    #cv2.destroyAllWindows()
    return X


def for_kinema_cal_1(L, H, q_all_before):
    """順運動学を計算する関数（調整中）
    ジョイント位置の円筒座標系表現を返す
    q_5,q_6は固定．

    引数1：リンク長さの配列．nd.array(6)．単位は[m]
    引数2：リンク高さの配列．nd.array(1)．単位は[m]
    引数3：ジョイント角度の横べくトル．nd.array((6))．単位は[°]．

    戻り値：全ジョイント位置（円筒座標系表現）を横連結した行列．nd.array((3,6))．単位は上の行から[m],[rad],[m]．
    """
    
    q_all = q_all_before * np.pi / 180  # [°]→[rad]に変換

    x_0 = np.array([[0], \
        [0], \
        [0]])
    x_1 = x_0 + \
        np.array([[0], \
        [0], \
        [L[0]]])
    x_2 = x_1 + \
        np.array([[0], \
        [q_all[0]], \
        [L[1]]])
    x_3 = x_2 + \
        np.array([[L[2] * np.cos(q_all[1])], \
        [0], \
        [L[2] * np.sin(q_all[1])]])
    x_4 = x_3 + \
        np.array([[L[3] * np.sin(q_all[1] + q_all[2])], \
        [0], \
        [-L[3] * np.cos(q_all[1] + q_all[2])]])
    x_7 = x_4 + \
        np.array([[-(L[4] + L[5] + L[6]) * np.cos(q_all[1] + q_all[2] + q_all[3]) + H[0] * np.sin(q_all[1] + q_all[2] + q_all[3])], \
        [0], \
        [-(L[4] + L[5] + L[6]) * np.sin(q_all[1] + q_all[2] + q_all[3]) - H[0] * np.cos(q_all[1] + q_all[2] + q_all[3])]])

    z = np.concatenate([x_0, x_1, x_2, x_3, x_4, x_7], 1)  # 横に結合
    return z


def coordinate_trans_from_cyl_to_car(x_all_cyl):
    """ジョイント位置を円筒座標系表現x_all_cylから直交座標系表現x_all_carへ座標変換する関数

    引数：全ジョイント位置（円筒座標系表現）を横連結した行列．nd.array((3,6))．単位は上の行から[m],[rad],[m]．

    戻り値：全ジョイント位置（直交座標系表現）を横連結した行列．nd.array((3,6))．単位は上の行から[m],[m],[m]．
    """

    x_all_car = np.zeros((3, 6))

    for i in np.arange(0, 6, 1):
        x_all_car[0, i] = x_all_cyl[0, i] * np.cos(x_all_cyl[1, i])
        x_all_car[1, i] = x_all_cyl[0, i] * np.sin(x_all_cyl[1, i])
        x_all_car[2, i] = x_all_cyl[2, i]

    z = x_all_car
    return z


def graph_arm_configration(L, H, q):
    """マニピュレータの姿勢をグラフ化する（調整中）

    引数1：リンク長さの配列．nd.array(6)．単位は[m]
    引数2：リンク高さの配列．nd.array(1)．単位は[m]
    引数3：

    戻り値：なし
    """

    joint_position_all_cyl = for_kinema_cal_1(L, H, q)
    print('円筒座標系でのジョイント座標は', joint_position_all_cyl)

    joint_position_all_car = np.zeros((3, 6))
    for i in range(0, 5):
        dx = coordinate_trans_from_cyl_to_car(joint_position_all_cyl[:, i:i+1])
        #print(dx)
        joint_position_all_car[0, i] = dx[0, 0]
        joint_position_all_car[1, i] = dx[1, 0]
        joint_position_all_car[2, i] = dx[2, 0]

    print(joint_position_all_car)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = joint_position_all_car[0, :]
    y = joint_position_all_car[1, :]
    z = joint_position_all_car[2, :]

    ax.plot(x, y, z, color = 'blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)

    plt.show()



# パラメーター

# リンク長さ[m]
L0 = 0.095
L1 = 0.016
L2 = 0.104
L3 = 0.099
L4 = 0.058
L5 = 0.03
L6 = 0.07
L = np.array([L0, L1, L2, L3, L4, L5, L6])  # ベクトル化
H45 = 0.028
H = np.array([H45])  # ベクトル化

# ジョイント角オフセット[°]
Q_1_OFFSET = -5
Q_2_OFFSET = 3
Q_3_OFFSET = 3
Q_4_OFFSET = 5
Q_5_OFFSET = 4
Q_6_OFFSET = 0  # 未測定?
JOINT_ANGLE_OFFSET = np.array([Q_1_OFFSET, Q_2_OFFSET, Q_3_OFFSET, Q_4_OFFSET, Q_5_OFFSET, Q_6_OFFSET])   # ベクトル化


# 本体

# 目標位置読み込み
# 固定値を指定[m]
"""
position_target_x = 0.27 * np.cos(np.pi / 6)
position_target_y = 0.27 * np.sin(np.pi / 6)
position_target_z = 0
position_target = np.array([[position_target_x], [position_target_y], [position_target_z]])  # ベクトル化

position_goal_x = 0.27 * np.cos(np.pi / 2)
position_goal_y = 0.27 * np.sin(np.pi / 2)
position_goal_z = 0
position_goal = np.array([[position_goal_x], [position_goal_y], [position_goal_z]])  # ベクトル化
"""

# opencvから取得[m]

position_opencv_xy = detect_mark(2) * 0.001 - 0.01 # mm→mに変換
#position_opencv_xy = detect_mark(2) * 0.001

hc = 0.05  # さいころの高さ
h = 1.165  # カメラの高さ

#position_target_x = position_opencv_xy[0, 0]
#position_target_y = position_opencv_xy[0, 1]
position_target_x = position_opencv_xy[0, 0] * (1 - hc / h)
position_target_y = (position_opencv_xy[0, 1] - 0.2025) * (1 - hc / h) + 0.2025
position_target_z = 0
position_target = np.array([[position_target_x], [position_target_y], [position_target_z]])  # ベクトル化
print('position_targetは',position_target)

position_goal_x = position_opencv_xy[1, 0]
position_goal_y = position_opencv_xy[1, 1]
position_goal_z = 0
position_goal = np.array([[position_goal_x], [position_goal_y], [position_goal_z]])  # ベクトル化
print('position_gpalは',position_goal)


# ロボットアームを動かす

# Arduino宣言
board = pyfirmata.Arduino('COM5')  # COMは適宜変更．'/dev/ttyACM0'を使うと毎回変更しないで済むらしい．ただなぜかエラーになるので使わない
print('ボード認識成功!')

# サーボ宣言．メソッド.writeの引数は[°]
joint_angle_1 = board.get_pin('d:11:s')
joint_angle_1.write(90 + JOINT_ANGLE_OFFSET[0])
print('サーボ1認識成功!')
time.sleep(0.01)
joint_angle_4 = board.get_pin('d:6:s')
joint_angle_4.write(90 + JOINT_ANGLE_OFFSET[3])
print('サーボ4認識成功!')
time.sleep(0.3)
joint_angle_3 = board.get_pin('d:9:s')
joint_angle_3.write(90 + JOINT_ANGLE_OFFSET[2])
print('サーボ3認識成功!') 
time.sleep(0.4)
joint_angle_2 = board.get_pin('d:10:s')
joint_angle_2.write(90 + + JOINT_ANGLE_OFFSET[1])
print('サーボ2認識成功!')
time.sleep(0.5)
joint_angle_5 = board.get_pin('d:5:s')
joint_angle_5.write(90 + JOINT_ANGLE_OFFSET[4])
print('サーボ5認識成功!')
time.sleep(0.01)
joint_angle_6 = board.get_pin('d:3:s')
joint_angle_6.write(90 + JOINT_ANGLE_OFFSET[5])
print('サーボ6認識成功!')

joint_angle_all = np.array([joint_angle_1, joint_angle_2, joint_angle_3, joint_angle_4, joint_angle_5, joint_angle_6])  # ベクトル化


#initial_configration(JOINT_ANGLE_OFFSET, joint_angle_all)
#time.sleep(300)


# 指先上げ下げの値
Z_HIGH = 0.2  # 上
Z_LOW = 0.05  # 下
wait_time = 2

# はじめに初期姿勢になる
initial_configration(JOINT_ANGLE_OFFSET, joint_angle_all)

time.sleep(wait_time)

# 把持対称のZ_HIGH上に指先を移動
print('把持対象物に移動中...')
change_configration(JOINT_ANGLE_OFFSET, position_target, joint_angle_all, Z_HIGH)

time.sleep(wait_time)

# クローを開く
print('クローを開きます...')
joint_angle_all[5].write(35)
time.sleep(wait_time)

# 指先を下げる
print('指先を下ろします...')
change_configration(JOINT_ANGLE_OFFSET, position_target, joint_angle_all, Z_LOW)

time.sleep(wait_time)

# クローを閉じる
print('クローを閉じます...')
joint_angle_all[5].write(80)

time.sleep(wait_time)

# 把持対称物をZ_HIGH持ち上げる
print('持ち上げます...')
change_configration(JOINT_ANGLE_OFFSET, position_target, joint_angle_all, Z_HIGH)

time.sleep(wait_time)

# 目標のZ_HIGH上に移動
print('目標に移動中...')
change_configration(JOINT_ANGLE_OFFSET, position_goal, joint_angle_all, Z_HIGH)

time.sleep(wait_time)

# 指先を下す
print('指先を下ろします...')
change_configration(JOINT_ANGLE_OFFSET, position_goal, joint_angle_all, Z_LOW)

time.sleep(wait_time)

# クローを開く
print('クローを開きます...')
joint_angle_all[5].write(35)

time.sleep(wait_time)

# 指先をZ_HIGH真上に移動
print('指先を上げます...')
change_configration(JOINT_ANGLE_OFFSET, position_goal, joint_angle_all, Z_HIGH)

time.sleep(wait_time)

# クローを閉じる
print('クローを閉じます...')
joint_angle_all[5].write(95)

time.sleep(wait_time)

# 初期姿勢に戻る
initial_configration(JOINT_ANGLE_OFFSET, joint_angle_all)

# 終了
del board
print('終了')




# 順運動学順運動学の検算
"""
q_1_test = 90
q_2_test = 90
q_3_test = 90
q_4_test = 90
q_5_test = 90  #　飾り
q_6_test = 0  # 飾り
q_all_test = np.array([q_1_test, q_2_test, q_3_test, q_4_test, q_5_test, q_6_test])  # ベクトル化

print('指定したジョイント角度は', q_all_test)
x_all_cyl = for_kinema_cal_1(L, H, q_all_test)
x_all_car = coordinate_trans_from_cyl_to_car(x_all_cyl)


print('順運動学解（円筒座標系）\n', x_all_cyl)
print('順運動学解（直交座標系）\n', x_all_car * 100, '[cm]')
"""


# 逆運動学，順運動学の検算
"""
position_test_x = 0.20 * np.cos(1 * np.pi / 2)
position_test_y = 0.20 * np.sin(1 * np.pi / 2)
position_test_z = 0.05
position_test = np.array([[position_test_x], [position_test_y], [position_test_z]])  # ベクトル化

print('指定位置\n', position_test)
q_all = inv_kinema_cal_3(L, H, position_test)
x_all_cyl = for_kinema_cal_1(L, H, q_all)
x_all_car = coordinate_trans_from_cyl_to_car(x_all_cyl)


print('\n逆運動学解（ジョイント角） = ', q_all)
print('\n逆運動学解（円筒座標系）\n', x_all_cyl)
print('\n逆運動学解（直交座標系）\n', x_all_car * 100, '[cm]')
print('\n指定位置\n', position_test * 100, '[cm]')
print('\n逆運動学解（直交座標系．指先）\n', x_all_car[:, 5:6] * 100, '[cm]')
print('\n指定位置と解の誤差（指定位置-指先位置）は\n', (position_test[:, 0:1] - x_all_car[:, 5:6]) * 100, '[cm]')
print('\n指定位置と解の誤差（ユークリッド距離）は', np.sqrt((position_test[0, 0] - x_all_car[0, 5:6]) ** 2 + (position_test[1, 0] - x_all_car[1, 5:6]) ** 2 + (position_test[2, 0] - x_all_car[2, 5:6]) ** 2) * 100, '[cm]．整数値[°]にしているので多少の誤差はしょうがない')
"""






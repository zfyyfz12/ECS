from train_eval import *

if __name__ == '__main__':

    for lyr_1_out, lyr_2_out, lyr_3_out in [(200,100,2),(400,200,2),(800,400,2),
                                           (400,200,100),(800,400,200),(1600,800,400)]:

        # run SCDNN
        print(f'SCDNN_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}')
        mask = np.load(f'./mask/elite_mask_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}.npy', allow_pickle=True).item()   
        train_and_eval('sc', mask, lyr_1_out, lyr_2_out, lyr_3_out, max_epoch = 1000, save_dnn=0)

        # run FCDNN
        print(f'FCDNN_{lyr_1_out}_{lyr_2_out}_{lyr_3_out}')
        train_and_eval('fc', 'none', lyr_1_out, lyr_2_out, lyr_3_out, max_epoch = 1000, save_dnn=0)


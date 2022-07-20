# echo "Bash version ${BASH_VERSION}..."
# degree=8
# for i in {00..14}
#   do
#     # mkdir /home/caixin/GazeData/mpii_448/p$i/face
#     python inference_gfpgan.py -i /home/caixin/GazeData/MPIIFaceGaze/Image/p$i/face -o /data1/GazeData/mpii_448/Image/p$i/face -v 1.3 -s 2 &
#     echo $i
#     [ `expr $i % $degree` -eq 7 ] && wait
#   done


echo "Bash version ${BASH_VERSION}..."
degree=8
for i in {00..14}
  do
    # mkdir /home/caixin/GazeData/mpii_448/p$i/face
    #CUDA_VISIBLE_DEVICES=1
    # python inference_gfpgan.py -i /home/caixin/GazeData/MPIIFaceGaze1/Image/p$i/face -o /data1/GazeData/mpii_448/xgaze_512_device0_5000/Image/p$i/face -v train_GFPGAN_xgaze_512_device0 -s 2



    CUDA_VISIBLE_DEVICES=`expr $i % 4` python inference_realesrgan.py -i /home/caixin/GazeData/MPIIFaceGaze/Image/p$i/face -o /data1/GazeData/MPIIRes/$v\_$iter/Image/p$i/face --version $1  --iter $2 --sothis $3  -g 0 &

    echo $i
    [ `expr $i % $degree` -eq 7 ] && [  $i -le 14 ] && wait
    echo hello_$i
  done

cp  -r /home/caixin/GazeData/MPIIFaceGaze/Label /data1/GazeData/MPIIRes/$1
# python process_res.py -v $v\_$iter

# degree=4
# for i in `seq 1 10`
# do
#     sleep 1 & # 提交到后台的任务
#     echo $i
#     [ `expr $i % $degree` -eq 0 ] && wait
# done
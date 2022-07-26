# echo "Bash version ${BASH_VERSION}..."
# degree=8
# for i in {00..14}
#   do
#     # mkdir /home/caixin/GazeData/mpii_448/p$i/face
#     python inference_gfpgan.py -i /home/caixin/GazeData/MPIIFaceGaze/Image/p$i/face -o /data1/GazeData/mpii_448/Image/p$i/face -v 1.3 -s 2 &
#     echo $i
#     [ `expr $i % $degree` -eq 7 ] && wait
#   done


# echo "Bash version ${BASH_VERSION}..."
# degree=6
# path=/data1/GazeData/gazecap/Image
# files=$(ls $path)
# for i in $files
#   do
#     # mkdir /home/caixin/GazeData/mpii_448/p$i/face
#     #CUDA_VISIBLE_DEVICES=1
#     # python inference_gfpgan.py -i /home/caixin/GazeData/MPIIFaceGaze1/Image/p$i/face -o /data1/GazeData/mpii_448/xgaze_512_device0_5000/Image/p$i/face -v train_GFPGAN_xgaze_512_device0 -s 2



#     CUDA_VISIBLE_DEVICES=`expr $i % 4` python inference_realesrgan.py -i /data1/GazeData/gazecap/Image/$i/face -o /data1/GazeData/GazeCap/$1\_$2/Image/$i/face --version $1  --iter $2 --sothis $3  --fp32 &

#     echo $i
#     [ `expr $i % $degree` -eq 4 ] && wait
#     echo hello_$i
#   done
# wait
# cp  -r /data1/GazeData/gazecap/Label /data1/GazeData/GazeCap/$1\_$2



# python process_res.py -v $v\_$iter

# degree=4
# for i in `seq 1 10`
# do
#     sleep 1 & # 提交到后台的任务
#     echo $i
#     [ `expr $i % $degree` -eq 0 ] && wait
# done


THREAD_NUM=6 #todo: revise me

# degree=6
i=0
path=/data1/GazeData/gazecap/Image
array=$(ls $path)

#定义描述符为9的FIFO管道
mkfifo tmp
exec 9<>tmp
rm -f tmp

#预先写入指定数量的空格符，一个空格符代表一个进程
for ((i=0;i<$THREAD_NUM;i++))
do
    echo >&9
done
i=0
for arg in ${array}; do
  #控制进程数：读出一个空格字符，如果管道为空，此处将阻塞
  i=`expr $i + 1`
  read -u9
  {

     #打印参数
     #echo ${arg}
     #此行代码指定任务提交方法

     echo $arg
     echo $i
     CUDA_VISIBLE_DEVICES=`expr $i % 4` python inference_realesrgan.py -i /data1/GazeData/gazecap/Image/$arg/face -o /data1/GazeData/GazeCap/$1\_$2/Image/$arg/face --version $1  --iter $2 --sothis $3  --fp32 --gpu 0

     #每执行完一个程序，睡眠3s
     sleep 3
     #控制进程数：一个任务完成后，写入一个空格字符到管道，新的任务将可以执行
     echo >&9
  }&
done
wait
echo "\n全部任务执行结束"

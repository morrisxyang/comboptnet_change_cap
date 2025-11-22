max_jobs=3
for i in {1..10}; do
  # 如果已经有3个在跑，就等
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 1
  done

  # 启动一个新任务（放到后台）
  python3 main.py experiments/knapsack/comboptnet.yaml &

  # 下一个任务启动前等 5 秒
  sleep 5
done

# 等所有后台任务结束
wait
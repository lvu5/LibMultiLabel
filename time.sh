peak=0
while kill -0 659723 2>/dev/null; do
    hwm=$(awk '/VmHWM/ {print $2}' /proc/659723/status)
    (( hwm > peak )) && peak=$hwm
    sleep 1
done
echo "Peak RAM: ${peak} kB"
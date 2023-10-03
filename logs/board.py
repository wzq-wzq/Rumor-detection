from tensorboard.backend.event_processing import event_accumulator

log_path="D:/Users/74148/wzq/code/python/project1/logs/events.out.tfevents."
mylog="finalbefore"
ea=event_accumulator.EventAccumulator(log_path+mylog+".LAPTOP-MB7530MT")
ea.Reload()
print(ea.scalars.Keys())
val_acc=ea.scalars.Items("val_accuracy")
acc=ea.scalars.Items("accuracy")
loss=ea.scalars.Items("loss")
step_length=int(val_acc[0].step)
print(val_acc)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 构建数据
#def model(x, p):
#    return x ** (2 * p + 1) / (1 + x ** (2 * p))
#x = np.linspace(0.75, 1.25, 201)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
# 设置图例标题大小
plt.rcParams['legend.title_fontsize'] = 9
fig, ax = plt.subplots(figsize=(8,6),dpi=100)
#colors = ["#0073C2","#EFC000","#868686","#CD534C","#7AA6DC","#003C67"]
#for p,c in zip([10, 15, 20, 30, 50, 100],colors):
#ax.plot([i.step/step_length for i in val_acc], [i.value for i in val_acc], color="#0073C2",label="val_acc")
ax.plot([i.step/step_length for i in acc], [i.value for i in acc], color="#0073C2",label="acc")
ax.plot([i.step/step_length for i in loss], [i.value for i in loss], color="#EFC000",label="loss")
#次刻度
yminorLocator = MultipleLocator(.25/2) #将此y轴次刻度标签设置为0.1的倍数
xminorLocator = MultipleLocator(1)
ax.yaxis.set_minor_locator(yminorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
#属性
ax.tick_params(which='major', length=5, width=1, direction='in', top='on',right="on")
ax.tick_params(which='minor', length=3, width=1,direction='in', top='on',right="on")
# axis label
ax.set_xlabel('epoch', fontsize=13,labelpad=5)
#ax.set_ylabel('acc or loss', fontsize=13,labelpad=5)
ax.set_ylabel('val_acc', fontsize=13,labelpad=5)
#网格
ax.grid(which='major',ls='-',alpha=.8,lw=.8)
#图例
ax.legend(fontsize=8,loc='upper left',title="Order")
#文本信息
ax.set_title("acc and loss",fontsize=14,pad=10)
#ax.set_title("val_acc",fontsize=14,pad=10)
ax.text(.87,.06,'',transform = ax.transAxes,
        ha='center', va='center',fontsize = 5)
plt.show()
# readme

電機所 碩一 N26091194 鄧立昌
電機所 碩一 N26092116 宋士莆

## Excution

直接在 Command line 執行 app.py 即可

```cmd=
python app.py
```

## Method

### Linear Regression

請參考 **model-ml-single_feature.ipynb**

使用 sklearn 的 LinearRegression 來作為 baseline model
雖然看起來趨勢大致相同，但實際上的預測能力很差。

![pred](https://i.imgur.com/i1xFJHv.png)

現在輸入頭一個數值，要預測接下來7天，
讓 model 根據預測值去 predict 下一數值，
這樣誤差將會越來越大。

![pred2](https://i.imgur.com/dyFiVHJ.png)

後續接進行其他 model 的測試。

### LSTM

請參考 **model-lstm-single_feature.ipynb**

嘗試使用 LSTM 進行預測，
把過去 1/3/5/7/30/60 天的資訊丟入 LSTM model 訓練。

```python=
class LSTM(nn.Module):
    def __init__(self ,input_size, hidden_size, num_layers, batchsize, device, out_dim):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = self.input_size, 
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers, 
                        batch_first = True)

        self.fc = nn.Linear(self.hidden_size, out_dim)
        
        self.hidden_cell = (torch.zeros(self.num_layers,batchsize,self.hidden_size).to(device),
                    torch.zeros(self.num_layers,batchsize,self.hidden_size).to(device))
    
    def forward(self, x):

        output, self.hidden_cell = self.lstm(x, self.hidden_cell)   

        x = self.fc(output)

        x = x[:,-1,:]

        x = x.unsqueeze(1)

        return x
```

但效果不是很好。
LSTM 都只會輸出一個固定的 output。

![](https://i.imgur.com/X146Lxj.png)

![](https://i.imgur.com/bUPrzFJ.png)

### CNN + Transformer encoder layer

請參考 **model-tf-single_feature.ipynb**

使用 conv1d + TransformerEncoderLayer 從過去 7/30 天的資料取特徵。

```python=
class CNN(nn.Module):
    def __init__(self ,seq_length, in_channels, kernel_size):
        super(CNN, self).__init__()

        self.seq_length = seq_length
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.feature = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=self.in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.tf_layer = nn.TransformerEncoderLayer(d_model=self.seq_length, nhead=4)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(seq_length*16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.feature(x)
        #print('1:',x.shape)
        x = self.tf_layer(x)
        #print('2:',x.shape)
        x = self.flatten(x)
        #print('3:',x.shape)
        x = self.fc(x)
        return x
```

MAR loss 如下:

![](https://i.imgur.com/Oib3eJ3.png)

模型一樣都預測同一數值。

![](https://i.imgur.com/HmHx5K8.png)

![](https://i.imgur.com/Qkb2cto.png)

![](https://i.imgur.com/EqBVFZB.png)

### randomforest

請參考 **forest**

使用randomforest評估趨勢線，覺得效果不比linea regression好便捨棄不用。

![](https://i.imgur.com/nmCDYFB.png)

### cross_val_predict

請參考 **cross_val**
使用skl內建的cross_val進行評估，以為效果會比linear好，後來發現沒有比較沒有傷害。

![](https://i.imgur.com/1QsYBay.png)

## 問題討論
一開始以為可以使用回歸的問題預測未來的備轉容量，但在測試完模型，朝優化模型方向前進時，發現好像苗頭不對，開始探討假設是否符合問題的根本，後來發現假設錯誤，之所以可以很好的擬和是因為我們並沒有將每筆輸入資料預測出來的結果當成輸出，而是用新的資料進行預測，看似很完美的擬和，實際上卻似乎沒有太大的幫助。
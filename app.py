import csv
# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    #df_training = pd.read_csv(args.training)
    #model = Model()
    #model.train(df_training)
    #df_result = model.predict(n_step=7)

    with open(args.output, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['data','operating_reserve(MW)'])
        date = 20210323
        for i in range(7):
            writer.writerow([str(date),3000])
            date += 1
    #df_result.to_csv(args.output, index=0)

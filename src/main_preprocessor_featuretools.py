from preprocessing import preprocessing as p

pre = p.Preprocessor()

pre.run_featuretools(export_to_csv = True)

print('featuretools matrix exported to CSV.')
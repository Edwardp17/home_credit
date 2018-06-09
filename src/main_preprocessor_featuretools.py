from preprocessing import preprocessing as p

pre = p.Preprocessor()

pre.run_feature_tools(export_to_csv = True)

print('featuretools matrix exported to CSV.')
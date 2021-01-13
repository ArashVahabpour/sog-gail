np.ones(path["raws"].shape[0]) * 2 + \
                        output_d.flatten() * 0.1 + \
                        np.sum(np.log(output_p) * path["encodes"], axis=1)

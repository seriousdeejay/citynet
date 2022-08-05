# function integrating the other functions
def preprocess(base_dir, outdir, language, key_words, remove_referral=True, overwrite_protection=True):
    """
        params:
            base_dir:             str;
                path to directory where extracted wikidump files can be found
            outdir:               str;
                path where processed files will be saved to (one file per multistream)
            language:             str;
                one of the following ['en', 'fr']
            key_words:            str, list;
                list of strings which must be included in article
            remove_referral:      bool, optional; default is True.
                if True referral pages will be removed
            overwrite_protection: bool, optional; default is True.
                if True confirmation will be asked before overwriting files
    """

    # establish that a valid language was chosen, if not abort function:
    lang_list = ['fr', 'en']
    if language not in lang_list:
        print(f"Invalid language was chosen. \n Please choose one of the following: {lang_list}")
        return

    # creating an output directory
    outdir = os.path.join(outdir, f'{language}wiki/')
#     outdir = f'../../../data/{language}wiki/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print(f'created directory at: {outdir}')
    else:
        pass

    # base input directory
#     base_dir = f"/Volumes/NIJMAN/THESIS/{language}wiki_extracted" # path/to/wikidump/extracted

    # list of multistream directories in base_dir
    dir_list = os.listdir(base_dir)


#     for directory in dir_list:
    for directory in tqdm(dir_list, total = len(dir_list), desc = "Progress Total"):
        dir_fp = os.path.join(base_dir, directory)
        if not directory.startswith("."):
            print(f"\nStarting preprocessing on: {dir_fp}")
            wikidump = read_stream(dir_fp) # read the files in the directory

            wikidump = split_dump(wikidump) # split the files
            wikidump = process_dump2(wikidump, key_words) # extract id, title, article

            df = pd.DataFrame(wikidump, columns = ['article_id', 'title', 'text'])

            if remove_referral:
                try:
                    df['length'] = [len(text.split()) for text in df.text]
                    df['length_title'] = [len(title.split()) for title in df.title]
                    n_referral = len(df[df.length == df.length_title])

                    df = df[['article_id', 'title', 'text']][df.length != df.length_title]
                    print(f"Removing {n_referral} referral pages")

                except:
                    print(f"Referral pages were not removed from multistream {directory}")
                    pass


            # saving the output

            outfile = f'{language}wikidump_{directory}.csv'
            outputfp = os.path.join(outdir, outfile)

            # call write_outputcsv function
            write_outputcsv(df, outputfp, overwrite_protection = overwrite_protection)
        else:
            print(f"Skipping: {dir_fp}")

    print(f"----------\nFiles in {base_dir} have been processed\n----------")

    return

import praw
from psaw import PushshiftAPI
import datetime as dt
import pandas as pd
import os


class subreddit_scraper:
    def __init__(self):
        self._reddit = praw.Reddit(
            client_id=os.environ.get("REDDIT_CLIENT_ID"),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
            user_agent=os.environ.get("REDDIT_USER_AGENT"),
        )

        self._api = PushshiftAPI(self._reddit)

    def scrape_comments(self, subreddit, start_date, end_date, verbose=True, caching=True):
        queue = self.queue_comments
        _extract = self._extract_comment_data
        postfix = "comments"
        _print = self._print_comment

        comments = self._scrape(subreddit, start_date, end_date, verbose, caching, queue, _extract, postfix, _print)
        return comments

    def scrape_submissions(self, subreddit, start_date, end_date, verbose=True, caching=True):
        queue = self.queue_submissions
        _extract = self._extract_submission_data
        postfix = "submissions"
        _print = self._print_submission

        submissions = self._scrape(subreddit, start_date, end_date, verbose, caching, queue, _extract, postfix, _print)
        return submissions

    def _scrape(self, subreddit, start_date, end_date, verbose, caching, queue, _extract, postfix, _print):
        _cache_size = 1000
        _print_freq = 100

        module_path = os.path.dirname(__file__)
        data_path = module_path + "/../data"

        start_date_unix = int(dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").timestamp())
        end_date_unix = int(dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").timestamp())
        data = []

        if caching:
            try:
                os.mkdir(module_path + "/cache")
            except:
                pass

        for i, fetched in enumerate(queue(subreddit, start_date_unix, end_date_unix)):
            extracted = _extract(fetched)
            data.append(extracted)

            if verbose and not (i + 1) % _print_freq:
                _print(i, fetched)

            if caching and not (i + 1) % _cache_size:
                pd.DataFrame(data[-_cache_size:]).to_csv(module_path + f"/cache/{subreddit}_{postfix}_cache_{i + 1}")

        data = pd.DataFrame(data)
        data.to_csv(data_path + f"/{subreddit}_{postfix}")

        return data

    def queue_submissions(self, subreddit, start_date_unix, end_date_unix):
        submissions_queue = self._api.search_submissions(
            before=end_date_unix,
            after=start_date_unix,
            subreddit=subreddit,
            limit=None
        )
        return submissions_queue

    def queue_comments(self, subreddit, start_date_unix, end_date_unix):
        comments_queue = self._api.search_comments(
            before=end_date_unix,
            after=start_date_unix,
            subreddit=subreddit,
            limit=None
        )
        return comments_queue

    @staticmethod
    def _extract_submission_data(submission):
        submission_data = {
            "id": submission.id,
            "date": submission.created_utc,
            "title": submission.title,
            "n_comments": submission.num_comments,
            "score": submission.score,
            "stickied": submission.stickied,
            "saved": submission.saved,
            "locked": submission.locked,
            "edited": submission.edited,
            "upratio": submission.upvote_ratio,
            "subreddit_shname": submission.subreddit.display_name
        }
        return submission_data

    @staticmethod
    def _extract_comment_data(comment):
        comment_data = {
            "id": comment.id,
            "parent_id": comment.parent_id,
            "date": comment.created_utc,
            "submission_id": comment.submission.id,
            "subreddit_shname": comment.subreddit.display_name,
            "score": comment.score,
            "saved": comment.saved,
            "edited": comment.edited,
            "body": comment.body
        }
        return comment_data

    @staticmethod
    def _print_comment(i, comment):
        print_time = dt.datetime.fromtimestamp(comment.created_utc)
        print_string = "Comment: {} | Date: {} | Body: {}"

        print_body_len = 36
        print_body = comment.body.replace("\n", "")
        if len(print_body) > print_body_len:
            print_body = print_body[:print_body_len - len("...")] + "..."
        else:
            print_body = print_body + (print_body_len - len(print_body)) * " "

        print(print_string.format(i+1, print_time, print_body), end="\r", flush=True)
        return

    @staticmethod
    def _print_submission(i, submission):
        print_date = dt.datetime.fromtimestamp(submission.created_utc)
        print_string = "Submission: {} | Title: {} | Date: {} | has comments: {}"

        print_title_len = 16
        print_title = submission.title[:print_title_len]
        if len(print_title) > print_title_len - 3:
            print_title = print_title + "..."

        print(print_string.format(i+1, print_title, print_date, submission.num_comments), end="\r", flush=True)
        return

########################################################################################################################
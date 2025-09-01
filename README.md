Pairwise Item Ranker

A lightweight web app to rank anything — food, outfits, songs, characters, designs, you name it — by making simple pairwise choices.
You don’t have to decide your full ranking upfront; just pick between two options at a time, and the app figures out the rest.

🚀 Why this?

Most people don’t actually know their favourites until they compare items side by side.
Lists feel abstract, but “Do I prefer A or B?” is concrete.

By repeating that process, the app builds a full, consistent ranking for you — often revealing surprising personal preferences.

⚙️ What it does

Works with any set of items (images used as the easiest representation).

2-at-a-time comparisons: winner stays, faces the next challenger.

Smart skipping: remembers past results and infers obvious outcomes (A > B, B > C ⇒ A > C).

Save & resume: keep your pairwise decisions in a JSON file.

Export rankings:

CSV (rank, filename, path)

Ranked ZIP (items renamed in rank order)

💡 How it helps

Turns vague “I kind of like this more…” feelings into a structured ranking.

Works for any collection:

your favourite meals 🍜

character lineups 🧙‍♂️

music covers 🎵

outfit designs 👗

… and more!

Helps you discover a true personal order of favourites, not just a guess.

📝 Notes

Runs fully locally — your files stay on your machine.

IDs are based on filenames — avoid duplicates with the same name.

Image format is just a container: each file = one “item.”

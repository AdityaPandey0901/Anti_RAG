# Empty on purpose: its presence makes pytest add the repo root to sys.path
# (rootdir "prepend" import mode) so tests can `import summarize_documents`
# and friends without a src-layout or installed package.

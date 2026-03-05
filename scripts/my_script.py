import insightface

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)


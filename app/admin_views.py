from app import app
from flask import request, render_template


@app.route("/admin/dashboard")
def admin_dashboard():
    return render_template("admin/index.html")

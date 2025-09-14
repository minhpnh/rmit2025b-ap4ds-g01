from flask import Blueprint, render_template, abort, request, jsonify
from ..models.product_repo import ProductRepo
from ..models.review_repo import ReviewRepo
from ..services.search_service import tokenize, match_query

bp = Blueprint("catalog", __name__)


@bp.get("/catalog")
def index():
    repo = ProductRepo()
    review_repo = ReviewRepo()
    q = request.args.get("q", "").strip()
    sel_div = request.args.get("division", "").strip()
    sel_dept = request.args.get("department", "").strip()
    sel_class = request.args.get("class", "").strip()
    sort = request.args.get("sort", "").strip()  # '', 'most_recommended'
    try:
        page = max(int(request.args.get("page", 1)), 1)
    except ValueError:
        page = 1
    try:
        per_page = max(min(int(request.args.get("per_page", 24)), 100), 1)
    except ValueError:
        per_page = 24

    # Build option lists from all products (simple approach)
    all_products_for_facets = repo.list()
    divisions = sorted({p.get("division", "") for p in all_products_for_facets if p.get("division")})
    departments = sorted({p.get("department", "") for p in all_products_for_facets if p.get("department")})
    classes = sorted({p.get("class", "") for p in all_products_for_facets if p.get("class")})

    # If no search and no sort by recommendation, paginate directly via DB
    if not q and sort != "most_recommended":
        products, total = repo.paginated(page=page, per_page=per_page,
                                         division=sel_div or None,
                                         department=sel_dept or None,
                                         class_=sel_class or None)
        # Recommended counts only for current page
        counts = review_repo.recommended_counts_for_items([str(p["id"]) for p in products])
        for p in products:
            p["recommended_count"] = counts.get(str(p["id"]), 0)
        has_next = (page * per_page) < total
        return render_template(
            "catalog/index.html",
            products=products,
            q=q,
            total=total,
            page=page,
            per_page=per_page,
            has_next=has_next,
            divisions=divisions,
            departments=departments,
            classes=classes,
            filters={
                "division": sel_div,
                "department": sel_dept,
                "class": sel_class,
                "sort": sort,
            },
        )

    # Otherwise: apply filters server-side and search/sort in memory, then paginate
    filtered = repo.filter(
        division=sel_div or None,
        department=sel_dept or None,
        class_=sel_class or None,
    )
    if q:
        q_tokens = tokenize(q)
        filtered = [p for p in filtered if match_query(p, q_tokens)]

    # Recommended counts (optimize: only for full set if sorting by most_recommended)
    if sort == "most_recommended":
        ids = [str(p["id"]) for p in filtered]
        counts = review_repo.recommended_counts_for_items(ids)
        for p in filtered:
            p["recommended_count"] = counts.get(str(p["id"]), 0)
        filtered.sort(key=lambda x: x.get("recommended_count", 0), reverse=True)
    total = len(filtered)

    # Paginate slice
    start = (page - 1) * per_page
    end = start + per_page
    page_items = filtered[start:end]

    # If we didn't compute counts, do page-only counts now
    if sort != "most_recommended":
        counts = review_repo.recommended_counts_for_items([str(p["id"]) for p in page_items])
        for p in page_items:
            p["recommended_count"] = counts.get(str(p["id"]), 0)

    has_next = end < total
    return render_template(
        "catalog/index.html",
        products=page_items,
        q=q,
        total=total,
        page=page,
        per_page=per_page,
        has_next=has_next,
        divisions=divisions,
        departments=departments,
        classes=classes,
        filters={
            "division": sel_div,
            "department": sel_dept,
            "class": sel_class,
            "sort": sort,
        },
    )


@bp.get("/catalog/data")
def data():
    """Return next page of products as HTML fragment for lazy loading."""
    q = request.args.get("q", "").strip()
    sel_div = request.args.get("division", "").strip()
    sel_dept = request.args.get("department", "").strip()
    sel_class = request.args.get("class", "").strip()
    sort = request.args.get("sort", "").strip()
    try:
        page = max(int(request.args.get("page", 1)), 1)
    except ValueError:
        page = 1
    try:
        per_page = max(min(int(request.args.get("per_page", 24)), 100), 1)
    except ValueError:
        per_page = 24

    # Delegate to index logic by reusing the same filtering/pagination (minimal duplication)
    # We mimic the same computation but only return the items HTML.
    repo = ProductRepo()
    review_repo = ReviewRepo()

    if not q and sort != "most_recommended":
        products, total = repo.paginated(page=page, per_page=per_page,
                                         division=sel_div or None,
                                         department=sel_dept or None,
                                         class_=sel_class or None)
        counts = review_repo.recommended_counts_for_items([str(p["id"]) for p in products])
        for p in products:
            p["recommended_count"] = counts.get(str(p["id"]), 0)
        has_next = (page * per_page) < total
    else:
        filtered = repo.filter(
            division=sel_div or None,
            department=sel_dept or None,
            class_=sel_class or None,
        )
        if q:
            q_tokens = tokenize(q)
            filtered = [p for p in filtered if match_query(p, q_tokens)]
        if sort == "most_recommended":
            ids = [str(p["id"]) for p in filtered]
            counts = review_repo.recommended_counts_for_items(ids)
            for p in filtered:
                p["recommended_count"] = counts.get(str(p["id"]), 0)
            filtered.sort(key=lambda x: x.get("recommended_count", 0), reverse=True)
        total = len(filtered)
        start = (page - 1) * per_page
        end = start + per_page
        products = filtered[start:end]
        if sort != "most_recommended":
            counts = review_repo.recommended_counts_for_items([str(p["id"]) for p in products])
            for p in products:
                p["recommended_count"] = counts.get(str(p["id"]), 0)
        has_next = end < total

    items_html = render_template("catalog/_items.html", products=products, q=q)
    return jsonify({
        "items_html": items_html,
        "has_next": has_next,
        "next_page": page + 1 if has_next else None,
        "total": total if 'total' in locals() else None
    })


@bp.get("/item/<id>")
def item(id):
    repo = ProductRepo()
    p = repo.get(id)
    if not p:
        return abort(404)
    reviews = ReviewRepo().by_item(id)
    return render_template("catalog/item_detail.html", p=p, reviews=reviews)

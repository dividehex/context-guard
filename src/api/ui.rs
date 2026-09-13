//! `GET /ui/conversations/{id}`: the explanation page. One self-contained
//! HTML file compiled into the binary; it reads the conversation id from its
//! own URL and calls the JSON API, so the server never renders conversation
//! text into HTML and the page works wherever the API is reachable.

use axum::http::header;
use axum::response::IntoResponse;

const PAGE: &str = include_str!("../../ui/conversation.html");

pub async fn conversation() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], PAGE)
}

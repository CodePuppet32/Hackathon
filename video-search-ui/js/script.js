(function(){

  //utils
  let create_element = function (element, content, c){
    let t = document.createElement(element);
    if (content){
      $(t).html(content);
    }
    if (c){
      $(t).addClass(c);
    }
    return $(t);
  };

  let set_html = function ($e, content, c){
    $e.html(content);
    if (c){
      $e.addClass(c);
    }
  };

  let _set_html = function (e, content, c){
    $(e).html(content);
    if (c){
      $(e).addClass(c);
    }
  };

  // init
  let index = 0;
  let url;
  let SEARCH_BASE_URL = "http://127.0.0.1:8080/search?";

  let $main_div = $("#main-div");

  let $form = $main_div.find("#search-form");
  let $search_text = $form.find('#search-text');
  let $spoiler_free_checkbox = $form.find("#spoilerFreeSearch");

  let $submit = $form.find('#search-submit');

  let $search_results = $main_div.find('#search-results');


  function show_data(response){
    console.log(response);
    let text = response.text;
    let lang = response.lang;

    $main_div.find('#text-ext').html(text);
    $main_div.find('#lang-ext').html(lang);

  }

  function error_handler(){
    console.log('Error');
  }

  function get_searching_header(search_text){
    let s = `Searching for '${search_text}'`;
    let $p = create_element("p", s, )

    return $p
  }

  function get_search_result_card(img_url, title, channel_name, video_url, is_spoiler){
    let $c = create_element("div", "", "search-result-card p-3 d-flex");

    let $img_div = create_element("div", '', "search-result-image-div");
    let $img = create_element("img", '', "search-result-thumbnail img-thumbnail rounded float-start");
    $img.attr("src", img_url);
    if (is_spoiler){
      $img.css("filter", "blur(20px)");
    }
    $img_div.append($img);

    $c.append($img_div);

    let $search_info_div = create_element("div", "", "ms-3");
    let $video_title = create_element("h4", title, "");
    let $channel_name = create_element("h6", channel_name, "");
    $search_info_div.append($video_title);
    $search_info_div.append($channel_name);
    $c.append($search_info_div);

    return $c;
  }

  function update_searching_header($s, search_text, need_spoiler_free_results){
    let s = `Showing regular results for '${search_text}'`;
    if (need_spoiler_free_results){
      s = `Showing Spoiler-free results for '${search_text}'`;

    }
    $s.html(s);
  }

  function show_error(){
    let $card = create_element("h2", "Unknown Error Occurred!", "");
    $search_results.append($card);
  }

  function show_search_results(response) {
    for (const obj of response.result){
      let is_spoiler = obj.is_spoiler;
      let display_title = obj.title;
      if (is_spoiler === 1) {
        display_title = obj.spoilerfree_title;
      }
      let $card = get_search_result_card(obj.thumbnails, display_title, "", obj.videoId, is_spoiler);
      $search_results.append($card);
    }
  }

  function update_search_results(text, spoiler_free){

    // remove while demo TODO
    //
    let url = `${SEARCH_BASE_URL}q=${text}&spoiler_free=${spoiler_free}&mock=false`;
    $.ajax({
      url: url,
      type: "GET",
      contentType: 'application/json;charset=UTF-8',
      success: show_search_results,
      error: show_error,
    });

  }

  function run_search(event){
    let search_text = $search_text.val();
    console.log(search_text);

    let need_spoiler_free_results = $spoiler_free_checkbox.prop("checked");
    console.log(need_spoiler_free_results)

    $search_results.html("");
    let $searching_header = get_searching_header(search_text);
    $search_results.append($searching_header);

    update_search_results(search_text, need_spoiler_free_results);

    // let $card = get_search_result_card("https://i.ytimg.com/vi/rdgCVPJkJPI/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLDD2J0n2sg9WsqjKvfVC_GTmzPtGQ", "V 1", "C 1", "Something");
    // $search_results.append($card);
    // let $card2 = get_search_result_card("https://i.ytimg.com/vi/rdgCVPJkJPI/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLDD2J0n2sg9WsqjKvfVC_GTmzPtGQ", "V 2", "C 2", "Something");
    // $search_results.append($card2);

    update_searching_header($searching_header, search_text, need_spoiler_free_results);
  }

  // event listeners
  $submit.on('click', run_search);

})();

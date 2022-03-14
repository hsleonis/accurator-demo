/*!
* Start Bootstrap - Freelancer v7.0.5 (https://startbootstrap.com/theme/freelancer)
* Copyright 2013-2021 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-freelancer/blob/master/LICENSE)
*/
//
// Scripts
//

window.addEventListener('DOMContentLoaded', event => {

    // Navbar shrink function
    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) {
            return;
        }
        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }

    };

    // Shrink the navbar 
    navbarShrink();

    // Shrink the navbar when page is scrolled
    document.addEventListener('scroll', navbarShrink);

    // Activate Bootstrap scrollspy on the main nav element
    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            offset: 72,
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

});

function ValidURL(str) {
  var regex = /(?:https?):\/\/(\w+:?\w*)?(\S+)(:\d+)?(\/|\/([\w#!:.?+=&%!\-\/]))?/;
  if(!regex .test(str)) {
    console.log("Please enter valid URL.");
    return false;
  } else {
    return true;
  }
}

function Progress(point) {
    // update progress bar

    val =  $(".progress-bar").attr('aria-valuenow');
    updated_val = parseInt(val) + point;
    $(".progress-bar").css('width', updated_val + '%').attr("aria-valuenow", updated_val);
}


function runModel(model_name, id, prval) {
    // run selected models
    $(".progress_output").append("<p class='success'>Running model:" + model_name + ".</p>");

    $.post( "/model", {name: model_name, sumid: id}, function(data) {
        console.log(data.msg);
        Progress(prval);
    }, "json");
}

function preprocessID(id, str, prval, checked) {
    // pre-process text

    $.post( "/preprocess", {uid: id, url: str}, function(data) {
        $(".progress_output").append("<p class='success'>Preprocessing article titled: '" + data.title + "' complete!</p>");
        Progress(prval);
        for(j=0; j<checked.length; j++) {
            runModel(checked[j], id, pr_val);
        }
    }, "json");
}

function parseURL(str, id, prval, checked) {
    // parse text from url

    $.post( "/parse", {url: str, uid: id}, function(data) {
        $(".progress_output").append("<p class='success'>Parsing " + data.url + " complete!</p>");
        Progress(prval);
        preprocessID(id, str, prval, checked);
    }, "json");
}

// summarize from url
$(".sum_btn").on("click", function(e) {
    e.preventDefault();
    tasks = 4;  // number of tasks

    // checked models
    checked = [];
    strs = "";
    var checks = document.querySelectorAll("input[type='checkbox']:checked");
    for(var i=0; i<checks.length; i++){
        tmp_model = checks[i].value;
        checked.push(tmp_model);
        strs += "<li>" + tmp_model + "</li>";
    }
    if(checks.length) {
        $(".output").html("<h6>Selected Models:</h6><ol>" + strs + "</ol>");
    } else {
        $(".output").html("<h6>No Models Selected!</h6>");
    }

    // urls
    url_list = [];
    str2 = "";
    var urls = document.querySelectorAll("input[name='urls[]']");
    for(var i=0; i<urls.length; i++){
        tmp_url = urls[i].value;
        if(ValidURL(tmp_url)) {
            url_list.push(tmp_url);             // push to list
            str2 += "<li>" + tmp_url + "</li>"; // prepare to display urls
        }
    }
    if(url_list.length) {
        $(".output2").html("<h6>URLs Added:</h6><ol>" + str2 + "</ol>");
    } else {
        $(".output2").html("<h6>No URLs Added!</h6>");
    }

    // progress bar
    $(".progress-container").show();

    // parse urls
    for(var i=0; i<url_list.length; i++){
        pr_val = 100/(tasks * url_list.length);
        parseURL(url_list[i], i, pr_val, checked);               // parse url
    }
});

// summarize from text
$(".text_btn").on("click", function(e) {
    e.preventDefault();

    // checked models
    checked = [];
    strs = "";
    var checks = document.querySelectorAll("input[type='checkbox']:checked");
    for(var i=0; i<checks.length; i++){
        tmp_model = checks[i].value;
        checked.push(tmp_model);
        strs += "<li>" + tmp_model + "</li>";
    }
    if(checks.length) {
        $(".output").html("<h6>Selected Models:</h6><ol>" + strs + "</ol>");
    } else {
        $(".output").html("<h6>No Models Selected!</h6>");
    }

    // text from text box
    txt = $('.text_sum_box').val();

    // text found
    if(txt.length) {
        $(".output2").html("<h6>Added article:</h6><ol>Text length: " + txt.length + "</ol>");
    } else {
        $(".output2").html("<h6>No Text Added!</h6>");
    }

    // progress bar
    $(".progress-container").show();

    tasks = 3;
    pr_val = 100/tasks;
    $.post( "/save_text", {txt: txt}, function(data) {
        $(".progress_output").append("<p class='success'>Saving custom text complete!</p>");
        Progress(pr_val);
        preprocessID(6, '#', pr_val, checked);
    }, "json");
});

function generate_table(res) {
    // hide table container
    $('.table-container').hide();

    // clear current html and append new
    $('.table-inner').empty().append(res.html);

    // initiate data table
    $('.table-inner #table_summarization').DataTable({
        pageLength: 10
    });

    // display star rating
    $(".db-rating").starRating({
        starSize: 22,
        callback: function(currentRating, $el){
            console.log($el.data('rating'));
            $.post( "/star_rating", {val: currentRating, uid: $el.data('uid')}, function(res) {
                console.log(res);
            }, "json");
        }
    });

    // display table container
    $('.table-container').show();
}

// table viewer
$(".view_btn").on("click", function(e) {
    e.preventDefault();

    // add notification
    $(".progress_output").append("<p class='success'>Collecting summaries...</p>");

    $.post( "/table_viewer", {}, function(res) {
        generate_table(res);
    }, "json");
});


// calculate metrics
$(".calc_btn").on("click", function(e) {
    e.preventDefault();

    // add notification
    $(".progress_output").append("<p class='success'>Calculating metrics...</p>");

    $.post( "/calc_metrics", {}, function(res) {
        generate_table(res);
    }, "json");
});

// edit feedback button
$(document).on("click", ".edit_btn" , function(e){
    e.preventDefault();
    $(this).parent().hide();    // .feedback-wrapper
    $(this).parent().next().show();
});

// feedback button
$(document).on("click", ".fd_btn" , function(e) {
    e.preventDefault();

    // add notification
    $(".progress_output").append("<p class='success'>Feedback sent.</p>");

    text = $(this).prev().val();
    uid = $(this).data('uid');
    form = $(this).parent();
    feedback_wrapper = $(this).parent().prev();

    $.post( "/save_feedback", {val: text, uid: uid}, function(res) {
        console.log(res.msg);
        feedback_wrapper.find('.feedback-text').text(text);
        feedback_wrapper.show();
        form.hide();
    }, "json");
});

// clear button
$(".clr_btn").on("click", function(e) {
    e.preventDefault();

    if(confirm("Are you sure you want to delete all data?")) {
        // add notification
        $(".progress_output").append("<p class='success'>Clearing all data...</p>");

        $.post( "/clear_table", {}, function(res) {
            if(res.success) {
                cls = "success";
            } else {
                cls = "error";
            }
            $(".progress_output").append("<p class='" + cls + "'>" + res.msg + "</p>");
            $(".view_btn").click();
        }, "json");
    }
});

// tabs
$(document).ready(function() {
    $('.nav-tabs a:first').tab('show');

    $('.nav-tabs a').click(function (e) {
        e.preventDefault();
        $(this).tab('show');
    });
});
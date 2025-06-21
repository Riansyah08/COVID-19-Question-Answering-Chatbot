$(document).ready(function () {
    function sendMessage() {
        let message = $("#messageInput").val().trim();
        if (message !== "") {
            $(".chat-box").append(`<div class="message user-message">${message}</div>`);
            let question = $("#messageInput").val();
            $("#messageInput").val(""); 
            $(".chat-box").scrollTop($(".chat-box")[0].scrollHeight);

            setTimeout(()=>botResponse(question), 500);
        }
    }

    async function botResponse(question) {
    try {
        const response = await $.ajax({
            url: "http://127.0.0.1:5000/ask",
            method: 'POST',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify({
                question: question
            })
        });
        let botReplies = [
            "I'm not sure how to respond to that.",
            "Sorry, I don't have an answer for that.",
            "I do not know how to respond to that.",
            "I'm not sure how to respond to that.",
        ];
        if (response.answer){
            $(".chat-box").append(`<div class="message bot-message">${response.answer}</div>`);
            $(".chat-box").scrollTop($(".chat-box")[0].scrollHeight);   
        }else{
            let randomReply = botReplies[Math.floor(Math.random() * botReplies.length)];
            $(".chat-box").append(`<div class="message bot-message">${randomReply}</div>`);
            $(".chat-box").scrollTop($(".chat-box")[0].scrollHeight);
        }

    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
    }
    
    $("#sendButton").off("click").on("click", sendMessage);
    
    $("#messageInput").off("keypress").on("keypress", function (e) {
        if (e.which === 13) { 
            e.preventDefault(); 
            sendMessage();
        }
    });

    for (let i = 0; i < 100; i++) {
        let star = $("<div class='star'></div>");
        let x = Math.random() * 100;
        let y = Math.random() * 100;
        let size = Math.random() * 3 + 2;
        star.css({ 
            position: "absolute",
            top: `${y}vh`, 
            left: `${x}vw`, 
            width: `${size}px`, 
            height: `${size}px`,
            background: "rgba(255,255,255,0.8)",
            borderRadius: "50%",
            animation: `twinkle ${Math.random() * 2 + 1}s infinite alternate`
        });
        $(".stars").append(star);
    }
});

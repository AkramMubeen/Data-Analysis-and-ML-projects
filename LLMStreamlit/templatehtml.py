css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}

.chat-message.user {
    background: linear-gradient(to right, #ff6b6b, #ffb74d); /* Gradient background for user messages */
}

.chat-message.bot {
    background: linear-gradient(to right, #00bcd4, #4caf50); /* Gradient background for bot messages */
    color: #fff; /* Text color for bot messages */
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #333; /* Text color for user messages */
}
</style>

'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/sw9sxFM/unnamed.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/mcYFq8f/nobita.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''